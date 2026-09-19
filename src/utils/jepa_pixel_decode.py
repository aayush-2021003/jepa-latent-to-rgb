"""Shared primitives for the isolated FactorJEPA-to-pixel experiment.

Gold standards:
  https://github.com/facebookresearch/vjepa2
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py

The experimental contract is deliberately narrow:
  * oracle training: full-video frozen FactorJEPA features -> Cosmos VAE latent;
  * predicted evaluation: freeze that decoder and replace only JEPA target tokens;
  * no token-type embedding tells the decoder which values were replaced.

This module does not import or modify the legacy stage1/stage2 implementation.
"""
import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TT
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

try:
    import lpips
except ImportError:
    lpips = None

try:
    from diffusers import AutoencoderKLWan
    from diffusers.utils import export_to_video
except ImportError:
    AutoencoderKLWan = None
    export_to_video = None

from utils.bootstrap import bootstrap_ci
from utils.data_download import iter_clips_parallel
from utils.training import build_student_predictor
from utils.video_io import decode_video_bytes


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for FactorJEPA/Cosmos decoding")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    return device


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(mapping)}")
    return mapping[name]


def read_clip_manifest(path: str) -> list[str]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text())
    if isinstance(payload, list):
        keys = payload
    elif isinstance(payload, dict):
        supported = [name for name in ("clip_keys", "saved_keys") if name in payload]
        if len(supported) != 1 or not isinstance(payload[supported[0]], list):
            raise ValueError(
                f"{manifest_path} must contain exactly one list field from clip_keys/saved_keys")
        keys = payload[supported[0]]
    else:
        raise ValueError(f"{manifest_path} must be a list or a supported manifest object")
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError(f"{manifest_path} contains no valid clip keys")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{manifest_path} contains duplicate clip keys")
    return keys


def source_video_id(clip_key: str) -> str:
    parts = Path(clip_key).parts
    if len(parts) < 2:
        raise ValueError(f"clip key has no source-video component: {clip_key!r}")
    return parts[-2]


def audit_source_disjoint(train_keys: list[str], val_keys: list[str], test_keys: list[str]) -> dict:
    split_sources = {
        "train": {source_video_id(key) for key in train_keys},
        "validation": {source_video_id(key) for key in val_keys},
        "test": {source_video_id(key) for key in test_keys},
    }
    overlaps = {
        "train_validation": sorted(split_sources["train"] & split_sources["validation"]),
        "train_test": sorted(split_sources["train"] & split_sources["test"]),
        "validation_test": sorted(split_sources["validation"] & split_sources["test"]),
    }
    if any(overlaps.values()):
        sizes = {name: len(values) for name, values in overlaps.items()}
        raise ValueError(f"source-video leakage across manifests: {sizes}")
    return {
        "clips": {"train": len(train_keys), "validation": len(val_keys), "test": len(test_keys)},
        "source_videos": {name: len(values) for name, values in split_sources.items()},
        "overlaps": {name: 0 for name in overlaps},
    }


def center_crop_clip(video_tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    video = video_tensor.float() / 255.0
    _, _, height, width = video.shape
    shorter = min(height, width)
    top = (height - shorter) // 2
    left = (width - shorter) // 2
    video = video[:, :, top:top + shorter, left:left + shorter]
    video = F.interpolate(video, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    return video.permute(1, 0, 2, 3).contiguous()


def augment_clip(video_tensor: torch.Tensor, augmentation_cfg: dict, crop_size: int) -> torch.Tensor:
    scale = tuple(augmentation_cfg["random_resize_scale"])
    ratio = tuple(augmentation_cfg["random_resize_ratio"])
    top, left, height, width = TT.RandomResizedCrop.get_params(video_tensor[0], scale=scale, ratio=ratio)
    video = video_tensor.float() / 255.0
    video = video[:, :, top:top + height, left:left + width]
    video = F.interpolate(video, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    if torch.rand(1).item() < float(augmentation_cfg["horizontal_flip"]):
        video = video.flip(-1)
    return video.permute(1, 0, 2, 3).contiguous()


def iter_video_batches(*, local_data: str, clip_keys: list[str], num_frames: int,
                       crop_size: int, batch_size: int, num_readers: int,
                       decode_workers: int, queue_timeout_seconds: int,
                       reader_join_timeout_seconds: int, training: bool,
                       augmentation_cfg: dict):
    """Yield [B,C,T,H,W] unit-range videos and fail if any requested clip is absent."""
    requested = set(clip_keys)
    if not requested:
        raise ValueError("clip manifest is empty")
    if batch_size <= 0 or num_readers <= 0 or decode_workers <= 0:
        raise ValueError("batch_size, num_readers and decode_workers must be positive")

    clip_q, stop_event, reader = iter_clips_parallel(
        local_data, subset_keys=requested, num_readers=num_readers)
    seen = set()
    pending_bytes = []
    pending_keys = []
    temp_dir = tempfile.mkdtemp(prefix="jepa_pixel_decode_")

    def decode_pending(pool, encoded, keys):
        futures = [
            pool.submit(decode_video_bytes, body, temp_dir, key, num_frames)
            for body, key in zip(encoded, keys)
        ]
        decoded = [future.result() for future in futures]
        valid = [(key, clip) for key, clip in zip(keys, decoded) if clip is not None]
        if len(valid) != len(keys):
            failed = [key for key, clip in zip(keys, decoded) if clip is None]
            raise RuntimeError(f"failed to decode {len(failed)} requested clips; first={failed[0]}")
        transform = augment_clip if training else center_crop_clip
        tensors = [
            transform(clip, augmentation_cfg, crop_size) if training else transform(clip, crop_size)
            for _, clip in valid
        ]
        return torch.stack(tensors, dim=0), [key for key, _ in valid]

    try:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            while True:
                item = clip_q.get(timeout=queue_timeout_seconds)
                if item is None:
                    break
                clip_key, mp4_bytes = item
                if clip_key in seen:
                    raise RuntimeError(f"duplicate clip emitted by TAR reader: {clip_key}")
                seen.add(clip_key)
                pending_bytes.append(mp4_bytes)
                pending_keys.append(clip_key)
                if len(pending_keys) == batch_size:
                    yield decode_pending(pool, pending_bytes, pending_keys)
                    pending_bytes, pending_keys = [], []
            if pending_keys:
                yield decode_pending(pool, pending_bytes, pending_keys)
    finally:
        stop_event.set()
        reader.join(timeout=reader_join_timeout_seconds)
        shutil.rmtree(temp_dir, ignore_errors=True)

    missing = sorted(requested - seen)
    extra = sorted(seen - requested)
    if missing or extra:
        raise RuntimeError(
            f"manifest/TAR mismatch: requested={len(requested)} seen={len(seen)} "
            f"missing={len(missing)} extra={len(extra)} first_missing={missing[:1]}"
        )


@dataclass
class CosmosVAE:
    vae: nn.Module
    latent_mean: torch.Tensor
    latent_scale: torch.Tensor
    temporal_scale: int
    dtype: torch.dtype


def load_cosmos_vae_only(cosmos_cfg: dict, device: torch.device) -> CosmosVAE:
    if AutoencoderKLWan is None:
        raise RuntimeError("diffusers with AutoencoderKLWan support is required")
    dtype = dtype_from_name(cosmos_cfg["dtype"])
    vae = AutoencoderKLWan.from_pretrained(
        cosmos_cfg["model_id"],
        subfolder=cosmos_cfg["vae_subfolder"],
        revision=cosmos_cfg["revision"],
        torch_dtype=dtype,
        token=os.environ["HF_TOKEN"] if "HF_TOKEN" in os.environ else None,
    )
    vae.to(device)
    vae.requires_grad_(False)
    vae.eval()
    if cosmos_cfg["enable_tiling"]:
        vae.enable_tiling()
    if cosmos_cfg["enable_slicing"]:
        vae.enable_slicing()
    latent_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).float()
    latent_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).float()
    temporal_scale = 2 ** sum(vae.temperal_downsample)
    print(
        f"Loaded Cosmos VAE only: {cosmos_cfg['model_id']} revision={cosmos_cfg['revision']} "
        f"z_dim={vae.config.z_dim} temporal_scale={temporal_scale}"
    )
    return CosmosVAE(
        vae=vae,
        latent_mean=latent_mean,
        latent_scale=1.0 / latent_std,
        temporal_scale=temporal_scale,
        dtype=dtype,
    )


def normalize_cosmos_latents(cosmos: CosmosVAE, latents: torch.Tensor) -> torch.Tensor:
    mean = cosmos.latent_mean.to(latents.device, latents.dtype)
    scale = cosmos.latent_scale.to(latents.device, latents.dtype)
    return (latents - mean) * scale


def denormalize_cosmos_latents(cosmos: CosmosVAE, latents: torch.Tensor) -> torch.Tensor:
    mean = cosmos.latent_mean.to(latents.device, latents.dtype)
    scale = cosmos.latent_scale.to(latents.device, latents.dtype)
    return latents / scale + mean


@torch.no_grad()
def encode_cosmos_latents(cosmos: CosmosVAE, raw_batch: torch.Tensor) -> torch.Tensor:
    vae_input = raw_batch * 2.0 - 1.0
    with torch.amp.autocast("cuda", dtype=cosmos.dtype, enabled=cosmos.dtype != torch.float32):
        encoded = cosmos.vae.encode(vae_input.to(cosmos.dtype))
    if not hasattr(encoded, "latent_dist"):
        raise RuntimeError("Cosmos VAE encode output has no latent_dist")
    latents = encoded.latent_dist.mode()
    return normalize_cosmos_latents(cosmos, latents).float()


def match_num_frames(video: torch.Tensor, target_num_frames: int, temporal_scale: int) -> torch.Tensor:
    if video.shape[2] == target_num_frames:
        return video
    video = torch.repeat_interleave(video, repeats=temporal_scale, dim=2)
    if video.shape[2] < target_num_frames:
        pad = video[:, :, -1:].repeat(1, 1, target_num_frames - video.shape[2], 1, 1)
        video = torch.cat([video, pad], dim=2)
    return video[:, :, :target_num_frames]


def decode_cosmos_latents(cosmos: CosmosVAE, normalized_latents: torch.Tensor,
                          num_frames: int, requires_grad: bool) -> torch.Tensor:
    latents = denormalize_cosmos_latents(cosmos, normalized_latents)
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        with torch.amp.autocast("cuda", dtype=cosmos.dtype, enabled=cosmos.dtype != torch.float32):
            decoded = cosmos.vae.decode(latents.to(cosmos.dtype), return_dict=False)[0]
    decoded = match_num_frames(decoded, num_frames, cosmos.temporal_scale)
    return ((decoded.float().clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        for prefix in ("module.", "backbone."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def _load_with_fraction(module: nn.Module, state_dict: dict, minimum_percent: float,
                        label: str) -> dict:
    message = module.load_state_dict(_strip_state_dict_prefixes(state_dict), strict=False)
    total = len(module.state_dict())
    loaded = total - len(message.missing_keys)
    percent = 100.0 * loaded / total
    if percent < minimum_percent:
        raise RuntimeError(
            f"{label} checkpoint load below threshold: loaded={loaded}/{total} ({percent:.2f}%) "
            f"minimum={minimum_percent:.2f}% missing={message.missing_keys[:8]} "
            f"unexpected={message.unexpected_keys[:8]}"
        )
    record = {
        "loaded_keys": loaded,
        "total_keys": total,
        "loaded_percent": percent,
        "missing_keys": list(message.missing_keys),
        "unexpected_keys": list(message.unexpected_keys),
    }
    print(
        f"Loaded {label}: {loaded}/{total} keys ({percent:.2f}%); "
        f"missing={len(message.missing_keys)} unexpected={len(message.unexpected_keys)}"
    )
    return record


def resolve_hf_checkpoint(repo_id: str, repo_type: str, filename: str) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        token=os.environ["HF_TOKEN"] if "HF_TOKEN" in os.environ else None,
    )


def sha256_file(path: str, chunk_size_bytes: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_factorjepa_checkpoint(*, checkpoint_path: str, model_cfg: dict, data_cfg: dict,
                               load_predictor: bool, device: torch.device,
                               factor_cfg: dict):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "student" not in payload:
        raise KeyError(f"{checkpoint_path} has no student state dict")
    if load_predictor and "predictor" not in payload:
        raise KeyError(f"{checkpoint_path} has no predictor state dict")

    build_model_cfg = dict(model_cfg)
    build_model_cfg["use_activation_checkpointing"] = factor_cfg["activation_checkpointing"]
    student, predictor = build_student_predictor(build_model_cfg, data_cfg)
    load_report = {
        "student": _load_with_fraction(
            student, payload["student"], float(model_cfg["min_student_load_pct"]), "student")
    }
    if load_predictor:
        load_report["predictor"] = _load_with_fraction(
            predictor, payload["predictor"], float(model_cfg["min_predictor_load_pct"]), "predictor")
    dtype = dtype_from_name(factor_cfg["dtype"])
    student.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if load_predictor:
        predictor.to(device=device, dtype=dtype).eval().requires_grad_(False)
    else:
        predictor = None
    if hasattr(student, "return_hierarchical"):
        student.return_hierarchical = model_cfg["n_output_distillation"] > 1
    return student, predictor, load_report


def normalize_for_factorjepa(raw_batch: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(raw_batch.device, raw_batch.dtype)
    std = IMAGENET_STD.to(raw_batch.device, raw_batch.dtype)
    return (raw_batch - mean) / std


def _require_feature_width(tokens: torch.Tensor, model_cfg: dict, label: str) -> torch.Tensor:
    allowed = {
        int(model_cfg["embed_dim"]),
        int(model_cfg["embed_dim"]) * int(model_cfg["n_output_distillation"]),
    }
    if tokens.ndim != 3 or tokens.shape[-1] not in allowed:
        raise RuntimeError(f"{label} has shape {tuple(tokens.shape)}; expected width in {sorted(allowed)}")
    return tokens.float()


@torch.no_grad()
def extract_oracle_features(raw_batch: torch.Tensor, student: nn.Module,
                            model_cfg: dict, factor_cfg: dict) -> torch.Tensor:
    dtype = dtype_from_name(factor_cfg["dtype"])
    normalized = normalize_for_factorjepa(raw_batch)
    kwargs = {"training": True} if model_cfg["n_output_distillation"] > 1 else {}
    with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        output = student(normalized, **kwargs)
    if isinstance(output, (tuple, list)):
        output = output[-1]
    return _require_feature_width(output, model_cfg, "oracle features")


def expand_mask(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if isinstance(mask, (tuple, list)):
        if len(mask) != 1:
            raise RuntimeError(f"expected one mask tensor, got {len(mask)}")
        mask = mask[0]
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2:
        raise RuntimeError(f"mask must be [B,N] or [N], got {tuple(mask.shape)}")
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    if mask.shape[0] != batch_size:
        raise RuntimeError(f"mask batch={mask.shape[0]} feature batch={batch_size}")
    return mask.long()


def stable_clip_seed(base_seed: int, clip_key: str, mask_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{mask_name}|{clip_key}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def sample_seeded_masks(mask_generator, *, batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        context_mask, target_mask = mask_generator(batch_size)
    finally:
        torch.random.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    return expand_mask(context_mask, batch_size), expand_mask(target_mask, batch_size)


@torch.no_grad()
def substitute_predicted_features(*, raw_batch: torch.Tensor, oracle_features: torch.Tensor,
                                  student: nn.Module, predictor: nn.Module,
                                  context_mask: torch.Tensor, target_mask: torch.Tensor,
                                  model_cfg: dict, factor_cfg: dict,
                                  mask_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = dtype_from_name(factor_cfg["dtype"])
    batch_size = raw_batch.shape[0]
    context_mask = expand_mask(context_mask.to(raw_batch.device), batch_size)
    target_mask = expand_mask(target_mask.to(raw_batch.device), batch_size)
    normalized = normalize_for_factorjepa(raw_batch)
    hierarchical = model_cfg["n_output_distillation"] > 1
    student_kwargs = {"training": True} if hierarchical else {}
    predictor_kwargs = {"mod": "video", "mask_index": mask_index} if hierarchical else {}
    with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        visible = student(normalized, masks=[context_mask], **student_kwargs)
        output = predictor(visible, [context_mask], [target_mask], **predictor_kwargs)
    predicted = output[0] if isinstance(output, tuple) else output
    predicted = _require_feature_width(predicted, model_cfg, "predicted features")
    if predicted.shape[1] != target_mask.shape[1]:
        raise RuntimeError(
            f"predicted token count={predicted.shape[1]} target-mask count={target_mask.shape[1]}"
        )
    if predicted.shape[-1] != oracle_features.shape[-1]:
        raise RuntimeError(
            f"predicted width={predicted.shape[-1]} oracle width={oracle_features.shape[-1]}"
        )
    hybrid = oracle_features.clone()
    batch_indices = torch.arange(batch_size, device=hybrid.device).unsqueeze(1).expand_as(target_mask)
    hybrid[batch_indices, target_mask] = predicted
    return hybrid, predicted


def token_grid_shape(model_cfg: dict, data_cfg: dict) -> tuple[int, int, int]:
    return (
        int(data_cfg["num_frames"]) // int(model_cfg["tubelet_size"]),
        int(model_cfg["crop_size"]) // int(model_cfg["patch_size"]),
        int(model_cfg["crop_size"]) // int(model_cfg["patch_size"]),
    )


def target_mask_to_pixels(target_mask: torch.Tensor, model_cfg: dict,
                          data_cfg: dict) -> torch.Tensor:
    grid = token_grid_shape(model_cfg, data_cfg)
    batch_size = target_mask.shape[0]
    flat = torch.zeros(batch_size, math.prod(grid), device=target_mask.device)
    flat.scatter_(1, target_mask, 1.0)
    mask = flat.view(batch_size, 1, *grid)
    mask = F.interpolate(
        mask,
        size=(int(data_cfg["num_frames"]), int(model_cfg["crop_size"]), int(model_cfg["crop_size"])),
        mode="nearest",
    )
    return mask


class HierarchicalLevelFusion(nn.Module):
    def __init__(self, embed_dim: int, levels: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.levels = levels
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(levels)])
        self.level_logits = nn.Parameter(torch.zeros(levels))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] == self.embed_dim:
            return self.norms[-1](tokens)
        expected = self.embed_dim * self.levels
        if tokens.shape[-1] != expected:
            raise RuntimeError(f"feature width={tokens.shape[-1]} expected={self.embed_dim} or {expected}")
        chunks = tokens.view(tokens.shape[0], tokens.shape[1], self.levels, self.embed_dim)
        normalized = torch.stack(
            [self.norms[index](chunks[:, :, index]) for index in range(self.levels)], dim=2)
        weights = torch.softmax(self.level_logits, dim=0).view(1, 1, self.levels, 1)
        return (normalized * weights).sum(dim=2)


class ResidualConv3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float, groups: int):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"decoder channels={channels} must be divisible by groups={groups}")
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size, padding=padding),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size, padding=padding),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.block(tensor)


class OracleJEPADecoder(nn.Module):
    """Decode a complete JEPA grid; predicted evaluation substitutes values only."""

    def __init__(self, *, embed_dim: int, levels: int, grid_shape: tuple[int, int, int],
                 latent_shape: tuple[int, int, int, int], decoder_cfg: dict):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.latent_shape = tuple(latent_shape)
        decoder_dim = int(decoder_cfg["dim"])
        self.level_fusion = HierarchicalLevelFusion(embed_dim, levels)
        self.input_projection = nn.Linear(embed_dim, decoder_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, math.prod(grid_shape), decoder_dim))
        self.blocks = nn.Sequential(*[
            ResidualConv3D(
                decoder_dim,
                int(decoder_cfg["kernel_size"]),
                float(decoder_cfg["dropout"]),
                int(decoder_cfg["group_norm_groups"]),
            )
            for _ in range(int(decoder_cfg["depth"]))
        ])
        self.output_norm = nn.GroupNorm(int(decoder_cfg["group_norm_groups"]), decoder_dim)
        self.output_projection = nn.Conv3d(decoder_dim, latent_shape[0], kernel_size=1)
        nn.init.trunc_normal_(self.position_embedding, std=float(decoder_cfg["position_init_std"]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected_tokens = math.prod(self.grid_shape)
        if features.shape[1] != expected_tokens:
            raise RuntimeError(f"decoder expected {expected_tokens} tokens, got {features.shape[1]}")
        tensor = self.input_projection(self.level_fusion(features)) + self.position_embedding
        tensor = tensor.transpose(1, 2).reshape(features.shape[0], -1, *self.grid_shape)
        tensor = self.blocks(tensor)
        _, latent_t, latent_h, latent_w = self.latent_shape
        tensor = F.interpolate(
            tensor, size=(latent_t, latent_h, latent_w), mode="trilinear", align_corners=False)
        return self.output_projection(self.output_norm(tensor))


def build_oracle_decoder(model_cfg: dict, data_cfg: dict, decode_cfg: dict,
                         latent_shape: tuple[int, int, int, int]) -> OracleJEPADecoder:
    return OracleJEPADecoder(
        embed_dim=int(model_cfg["embed_dim"]),
        levels=int(model_cfg["n_output_distillation"]),
        grid_shape=token_grid_shape(model_cfg, data_cfg),
        latent_shape=latent_shape,
        decoder_cfg=decode_cfg["decoder"],
    )


def load_oracle_decoder(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "decoder_state_dict", "model_cfg", "data_cfg", "decode_cfg", "latent_shape", "grid_shape",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"oracle decoder checkpoint missing fields: {missing}")
    decoder = build_oracle_decoder(
        payload["model_cfg"], payload["data_cfg"], payload["decode_cfg"],
        tuple(payload["latent_shape"]),
    ).to(device)
    if tuple(decoder.grid_shape) != tuple(payload["grid_shape"]):
        raise RuntimeError("decoder checkpoint grid shape does not match reconstructed decoder")
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    decoder.eval().requires_grad_(False)
    return decoder, payload


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, epsilon: float,
                mask: torch.Tensor = None) -> torch.Tensor:
    values = torch.sqrt((prediction - target).pow(2) + epsilon * epsilon)
    if mask is None:
        return values.mean()
    expanded = mask.expand(-1, prediction.shape[1], -1, -1, -1)
    denominator = expanded.sum().clamp_min(1.0)
    return (values * expanded).sum() / denominator


def spatial_gradient_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
    pred_dx = prediction[..., 1:] - prediction[..., :-1]
    target_dx = target[..., 1:] - target[..., :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return charbonnier(pred_dx, target_dx, epsilon) + charbonnier(pred_dy, target_dy, epsilon)


def temporal_difference_loss(prediction: torch.Tensor, target: torch.Tensor,
                             epsilon: float) -> torch.Tensor:
    return charbonnier(
        prediction[:, :, 1:] - prediction[:, :, :-1],
        target[:, :, 1:] - target[:, :, :-1],
        epsilon,
    )


class LPIPSComputer:
    def __init__(self, metric_cfg: dict, device: torch.device):
        self.enabled = bool(metric_cfg["lpips_enabled"])
        self.max_frames = int(metric_cfg["lpips_max_frames"])
        if self.enabled:
            if lpips is None:
                raise RuntimeError("lpips package is required because lpips_enabled=true")
            self.model = lpips.LPIPS(net=metric_cfg["lpips_net"]).to(device).eval()
            self.model.requires_grad_(False)
        else:
            self.model = None

    def per_clip(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.full((prediction.shape[0],), torch.nan, device=prediction.device)
        frame_count = prediction.shape[2]
        if frame_count > self.max_frames:
            indices = torch.linspace(
                0, frame_count - 1, steps=self.max_frames, device=prediction.device).round().long()
            prediction = prediction.index_select(2, indices)
            target = target.index_select(2, indices)
        pred_frames = prediction.permute(0, 2, 1, 3, 4).flatten(0, 1) * 2.0 - 1.0
        target_frames = target.permute(0, 2, 1, 3, 4).flatten(0, 1) * 2.0 - 1.0
        values = self.model(pred_frames.float(), target_frames.float()).view(prediction.shape[0], -1)
        return values.mean(dim=1)


def per_clip_pixel_metrics(prediction: torch.Tensor, target: torch.Tensor,
                           metric_cfg: dict, lpips_computer: LPIPSComputer,
                           mask: torch.Tensor = None) -> dict[str, torch.Tensor]:
    difference = prediction.float() - target.float()
    if mask is None:
        mse = difference.pow(2).flatten(1).mean(dim=1)
        mae = difference.abs().flatten(1).mean(dim=1)
    else:
        expanded = mask.expand(-1, prediction.shape[1], -1, -1, -1)
        denominator = expanded.flatten(1).sum(dim=1).clamp_min(1.0)
        mse = (difference.pow(2) * expanded).flatten(1).sum(dim=1) / denominator
        mae = (difference.abs() * expanded).flatten(1).sum(dim=1) / denominator
    psnr = -10.0 * torch.log10(mse.clamp_min(float(metric_cfg["psnr_epsilon"])))
    temporal_pred = prediction[:, :, 1:] - prediction[:, :, :-1]
    temporal_target = target[:, :, 1:] - target[:, :, :-1]
    temporal_l1 = (temporal_pred - temporal_target).abs().flatten(1).mean(dim=1)
    result = {
        "pixel_mse": mse,
        "pixel_mae": mae,
        "psnr_db": psnr,
        "temporal_difference_l1": temporal_l1,
    }
    if mask is None:
        result["lpips"] = lpips_computer.per_clip(prediction, target)
    return result


def per_clip_latent_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    difference = prediction.float() - target.float()
    return {
        "latent_mse": difference.pow(2).flatten(1).mean(dim=1),
        "latent_mae": difference.abs().flatten(1).mean(dim=1),
    }


def compute_oracle_training_loss(*, predicted_latent: torch.Tensor, target_latent: torch.Tensor,
                                 raw_batch: torch.Tensor, cosmos: CosmosVAE,
                                 decode_cfg: dict, lpips_computer: LPIPSComputer,
                                 global_step: int) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = decode_cfg["loss"]
    epsilon = float(loss_cfg["charbonnier_epsilon"])
    latent_l1 = charbonnier(predicted_latent.float(), target_latent.float(), epsilon)
    latent_mse = F.mse_loss(predicted_latent.float(), target_latent.float())
    total = float(loss_cfg["latent_l1_weight"]) * latent_l1
    total = total + float(loss_cfg["latent_mse_weight"]) * latent_mse
    values = {
        "latent_l1": float(latent_l1.detach()),
        "latent_mse": float(latent_mse.detach()),
        "rgb_l1": 0.0,
        "spatial_gradient": 0.0,
        "temporal_difference": 0.0,
        "lpips": 0.0,
    }
    decoded_interval = int(loss_cfg["decoded_loss_every_steps"])
    decoded_weights = sum(float(loss_cfg[name]) for name in (
        "rgb_l1_weight", "spatial_gradient_weight", "temporal_difference_weight", "lpips_weight"))
    run_decoded = decoded_interval > 0 and global_step % decoded_interval == 0 and decoded_weights > 0
    if run_decoded:
        predicted_video = decode_cosmos_latents(
            cosmos, predicted_latent, raw_batch.shape[2], requires_grad=True)
        rgb_l1 = charbonnier(predicted_video, raw_batch, epsilon)
        spatial = spatial_gradient_loss(predicted_video, raw_batch, epsilon)
        temporal = temporal_difference_loss(predicted_video, raw_batch, epsilon)
        total = total + float(loss_cfg["rgb_l1_weight"]) * rgb_l1
        total = total + float(loss_cfg["spatial_gradient_weight"]) * spatial
        total = total + float(loss_cfg["temporal_difference_weight"]) * temporal
        values.update({
            "rgb_l1": float(rgb_l1.detach()),
            "spatial_gradient": float(spatial.detach()),
            "temporal_difference": float(temporal.detach()),
        })
        if float(loss_cfg["lpips_weight"]) > 0:
            lpips_loss = lpips_computer.per_clip(predicted_video, raw_batch).mean()
            total = total + float(loss_cfg["lpips_weight"]) * lpips_loss
            values["lpips"] = float(lpips_loss.detach())
    values["total"] = float(total.detach())
    values["decoded_losses_ran"] = bool(run_decoded)
    return total, values


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_records(path: Path, unique_fields: tuple[str, ...]) -> list[dict]:
    if not path.exists():
        return []
    records = []
    identities = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = [field for field in unique_fields if field not in record]
        if missing:
            raise ValueError(f"{path}:{line_number} missing identity fields {missing}")
        identity = tuple(record[field] for field in unique_fields)
        if identity in identities:
            raise ValueError(f"{path}:{line_number} duplicate record identity {identity}")
        identities.add(identity)
        records.append(record)
    return records


def write_csv(path: Path, records: list[dict]):
    if not records:
        raise ValueError("cannot write empty CSV")
    columns = sorted({key for record in records for key in record})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def summarize_records(records: list[dict], metric_names: list[str], bootstrap_cfg: dict) -> dict:
    if not records:
        raise ValueError("cannot summarize zero records")
    summary = {"n_records": len(records), "metrics": {}}
    for name in metric_names:
        values = np.asarray([record[name] for record in records], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if values.size == 1:
            mean = float(values[0])
            interval = {"mean": mean, "ci_lo": mean, "ci_hi": mean, "ci_half": 0.0}
        else:
            interval = bootstrap_ci(
                values,
                n_boot=int(bootstrap_cfg["iterations"]),
                ci=float(bootstrap_cfg["confidence_level"]),
                seed=int(bootstrap_cfg["seed"]),
            )
        interval["n"] = int(values.size)
        summary["metrics"][name] = interval
    return summary


def tensor_to_pil_frames(video: torch.Tensor) -> list:
    array = (video.detach().float().clamp(0, 1) * 255.0).byte()
    array = array.permute(1, 2, 3, 0).cpu().numpy()
    return [Image.fromarray(frame) for frame in array]


def export_tensor_video(video: torch.Tensor, output_path: Path, fps: int):
    if export_to_video is None:
        raise RuntimeError("diffusers is required to export MP4 videos")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(tensor_to_pil_frames(video), str(output_path), fps=fps)


def export_comparison_video(videos: list[torch.Tensor], labels: list[str],
                            output_path: Path, fps: int, label_cfg: dict):
    if len(videos) != len(labels) or not videos:
        raise ValueError("comparison videos and labels must have the same non-zero length")
    frames_per_video = [tensor_to_pil_frames(video) for video in videos]
    frame_count = len(frames_per_video[0])
    if any(len(frames) != frame_count for frames in frames_per_video):
        raise RuntimeError("comparison videos have different frame counts")
    font = ImageFont.load_default(size=int(label_cfg["font_size"]))
    combined = []
    for frame_index in range(frame_count):
        panels = []
        for label, frames in zip(labels, frames_per_video):
            panel = frames[frame_index].copy()
            draw = ImageDraw.Draw(panel)
            draw.rectangle(
                (0, 0, panel.width, int(label_cfg["bar_height"])),
                fill=tuple(label_cfg["background_rgb"]),
            )
            draw.text(
                (int(label_cfg["left_padding"]), int(label_cfg["top_padding"])),
                label,
                fill=tuple(label_cfg["text_rgb"]),
                font=font,
            )
            panels.append(panel)
        canvas = panels[0]
        for panel in panels[1:]:
            joined = Image.new("RGB", (canvas.width + panel.width, canvas.height))
            joined.paste(canvas, (0, 0))
            joined.paste(panel, (canvas.width, 0))
            canvas = joined
        combined.append(canvas)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(combined, str(output_path), fps=fps)


def atomic_torch_save(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
