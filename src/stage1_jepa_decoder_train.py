import argparse
import gc
import hashlib
import json
import math
import os
import queue
import random
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_ENDPOINT"] = os.environ.get("FACTORJEPA_HF_ENDPOINT", "https://huggingface.co")
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = os.environ["HF_ENDPOINT"]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

from utils.cache_policy import add_cache_policy_arg, resolve_cache_policy_interactive, wipe_output_dir
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
from utils.config import check_gpu, get_pipeline_config, load_merged_config, load_subset
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.data_paths import find_video_shards
from utils.gpu_batch import cuda_cleanup
from utils.progress import make_pbar
from utils.training import build_mask_generators, build_student_predictor, load_config
from utils.video_io import decode_video_bytes


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
DEFAULT_SOURCE_HF_REPO = "anonymousML123/factorjepa-outputs"
DEFAULT_SOURCE_HF_REPO_TYPE = "dataset"
DEFAULT_SOURCE_HF_FILENAME = (
    "outputs/full/vjepa_2_1_vitg_1B/train/"
    "m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt"
)
DEFAULT_SOURCE_CKPT = None
DENSEWORLD_DATASET_ID = "anonymousML123/denseworld-115k_archive"
CHECKPOINT_LATEST = "stage1_jepa_decoder_latest.pt"
CHECKPOINT_FINAL = "stage1_jepa_decoder.pt"
CHECKPOINT_STEP_TEMPLATE = "stage1_jepa_decoder_step_{step:07d}.pt"
LOSS_KEYS = [
    "loss_total",
    "loss_latent_l1",
    "loss_latent_mse",
    "loss_rgb_l1",
    "loss_spatial_gradient",
    "loss_temporal_difference",
    "loss_lpips",
]


@dataclass
class CosmosVAEHandle:
    """The only Cosmos component needed by Stage-1 and Stage-2."""

    vae: nn.Module
    latent_mean: torch.Tensor
    latent_scale: torch.Tensor
    temporal_scale: int
    dtype: torch.dtype


def torch_load_cpu(path: Path, *, weights_only: bool = False):
    """Memory-map checkpoints so CPU pages are loaded on demand when supported."""
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=weights_only)


def print_cuda_memory(label: str):
    if not torch.cuda.is_available():
        return
    gib = 1024 ** 3
    print(
        f"  CUDA memory after {label}: "
        f"allocated={torch.cuda.memory_allocated() / gib:.2f} GiB, "
        f"reserved={torch.cuda.memory_reserved() / gib:.2f} GiB"
    )


def normalize_for_jepa(raw_batch_unit_range: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(raw_batch_unit_range.device)
    std = IMAGENET_STD.to(raw_batch_unit_range.device)
    return (raw_batch_unit_range - mean) / std


def augment_clip_consistent(video_tensor: torch.Tensor, cfg_aug: dict, crop_size: int) -> torch.Tensor:
    import torchvision.transforms as TT

    scale = cfg_aug["random_resize_scale"]
    ratio = cfg_aug["random_resize_ratio"]
    i, j, h, w = TT.RandomResizedCrop.get_params(video_tensor[0], scale=scale, ratio=ratio)
    video = video_tensor.float() / 255.0
    video = video[:, :, i:i + h, j:j + w]
    video = F.interpolate(video, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    if torch.rand(1).item() < cfg_aug["horizontal_flip"]:
        video = video.flip(-1)
    return video


def resize_center_crop(video_tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    video = video_tensor.float() / 255.0
    _, _, h, w = video.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    video = video[:, :, top:top + side, left:left + side]
    video = F.interpolate(video, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    return video.permute(1, 0, 2, 3).contiguous()


def clip_key_from_json(json_bytes: bytes) -> str:
    meta = json.loads(json_bytes)
    return "{}/{}/{}".format(
        meta.get("section", ""), meta.get("video_id", ""), meta.get("source_file", ""))


def ordered_clip_keys_from_tar(tar_path: Path) -> list[str]:
    keys = []
    with tarfile.open(tar_path, "r") as tar:
        entries = {}
        for member in tar.getmembers():
            base = member.name.rsplit(".", 1)[0]
            ext = member.name.rsplit(".", 1)[-1] if "." in member.name else ""
            entries.setdefault(base, {})[ext] = member
        for parts in entries.values():
            if "json" not in parts or "mp4" not in parts:
                continue
            f = tar.extractfile(parts["json"])
            if f is None:
                continue
            try:
                keys.append(clip_key_from_json(f.read()))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return keys


def ordered_clip_keys_from_tars(local_data: str, max_tar_files: int,
                                source_num_samples: int) -> tuple[list[str], list[Path]]:
    tar_files = find_video_shards(local_data)
    if not tar_files:
        raise FileNotFoundError(f"no video TAR shards found under {local_data}")
    if max_tar_files <= 0:
        raise ValueError(f"max_tar_files must be positive, got {max_tar_files}")
    selected_tars = tar_files[:max_tar_files]
    if len(selected_tars) != max_tar_files:
        raise RuntimeError(
            f"requested the first {max_tar_files} TARs, but only found {len(selected_tars)}"
        )
    keys = []
    for tar_path in selected_tars:
        keys.extend(ordered_clip_keys_from_tar(tar_path))
    if len(keys) < source_num_samples:
        raise RuntimeError(
            f"the first {max_tar_files} TARs contain {len(keys)} usable clips, "
            f"expected at least {source_num_samples}"
        )
    source_keys = keys[:source_num_samples]
    if len(set(source_keys)) != len(source_keys):
        raise RuntimeError("duplicate clip keys found in the requested split source")
    return source_keys, selected_tars


def _keys_sha256(keys: list[str]) -> str:
    payload = "\n".join(keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative_tar_names(local_data: str, tar_files: list[Path]) -> list[str]:
    root = Path(local_data).resolve()
    names = []
    for tar_path in tar_files:
        resolved = tar_path.resolve()
        try:
            names.append(str(resolved.relative_to(root)))
        except ValueError:
            names.append(resolved.name)
    return names


def _validate_random_split_manifest(manifest: dict, source_keys: list[str],
                                    tar_names: list[str], split_cfg: dict):
    required = {
        "schema_version", "seed", "max_tar_files", "source_num_samples",
        "test_fraction", "validation_num_samples", "tar_files",
        "source_clip_keys_sha256", "train_clip_keys", "validation_clip_keys",
        "test_clip_keys",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise RuntimeError(f"split manifest is missing fields: {missing}")

    expected = {
        "schema_version": 1,
        "seed": int(split_cfg["seed"]),
        "max_tar_files": int(split_cfg["max_tar_files"]),
        "source_num_samples": int(split_cfg["source_num_samples"]),
        "test_fraction": float(split_cfg["test_fraction"]),
        "validation_num_samples": int(split_cfg["validation_num_samples"]),
        "tar_files": tar_names,
        "source_clip_keys_sha256": _keys_sha256(source_keys),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"split manifest {key}={manifest.get(key)!r} does not match "
                f"the current dataset/config value {value!r}. Use a new manifest path "
                "when changing the split configuration."
            )

    train_keys = list(manifest["train_clip_keys"])
    val_keys = list(manifest["validation_clip_keys"])
    test_keys = list(manifest["test_clip_keys"])
    split_lists = {"train": train_keys, "validation": val_keys, "test": test_keys}
    for name, keys in split_lists.items():
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate clip keys inside the {name} split")
    if set(train_keys) & set(val_keys) or set(train_keys) & set(test_keys) or set(val_keys) & set(test_keys):
        raise RuntimeError("train, validation, and test splits are not disjoint")
    if set(train_keys) | set(val_keys) | set(test_keys) != set(source_keys):
        raise RuntimeError("split manifest keys do not exactly cover the selected source clips")

    expected_test = int(round(len(source_keys) * float(split_cfg["test_fraction"])))
    if len(test_keys) != expected_test:
        raise RuntimeError(f"test split has {len(test_keys)} clips, expected {expected_test}")
    if len(val_keys) != int(split_cfg["validation_num_samples"]):
        raise RuntimeError(
            f"validation split has {len(val_keys)} clips, "
            f"expected {split_cfg['validation_num_samples']}"
        )


def load_or_create_random_split(local_data: str, split_cfg: dict,
                                manifest_path: str = None,
                                require_existing: bool = False
                                ) -> tuple[set[str], list[str], list[str], dict, Path]:
    source_num_samples = int(split_cfg["source_num_samples"])
    max_tar_files = int(split_cfg["max_tar_files"])
    source_keys, tar_files = ordered_clip_keys_from_tars(
        local_data, max_tar_files, source_num_samples)
    tar_names = _relative_tar_names(local_data, tar_files)
    path = Path(manifest_path or split_cfg["manifest_path"]).expanduser().resolve()

    if path.exists():
        manifest = json.loads(path.read_text())
        action = "Loaded"
    else:
        if require_existing:
            raise FileNotFoundError(
                f"shared split manifest does not exist: {path}. "
                "Run Stage-1 first so it creates the manifest."
            )
        shuffled = list(source_keys)
        random.Random(int(split_cfg["seed"])).shuffle(shuffled)
        test_count = int(round(source_num_samples * float(split_cfg["test_fraction"])))
        validation_count = int(split_cfg["validation_num_samples"])
        if test_count + validation_count >= source_num_samples:
            raise ValueError("test and validation splits leave no training samples")
        test_keys = shuffled[:test_count]
        val_keys = shuffled[test_count:test_count + validation_count]
        train_keys = shuffled[test_count + validation_count:]
        manifest = {
            "schema_version": 1,
            "seed": int(split_cfg["seed"]),
            "max_tar_files": max_tar_files,
            "source_num_samples": source_num_samples,
            "test_fraction": float(split_cfg["test_fraction"]),
            "validation_num_samples": validation_count,
            "tar_files": tar_names,
            "source_clip_keys_sha256": _keys_sha256(source_keys),
            "train_clip_keys": train_keys,
            "validation_clip_keys": val_keys,
            "test_clip_keys": test_keys,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp_path, path)
        action = "Created"

    _validate_random_split_manifest(manifest, source_keys, tar_names, split_cfg)
    manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {
        "split_type": "deterministic_random",
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
        "seed": manifest["seed"],
        "tar_files": manifest["tar_files"],
        "source_num_samples": manifest["source_num_samples"],
        "train_num_samples": len(manifest["train_clip_keys"]),
        "validation_num_samples": len(manifest["validation_clip_keys"]),
        "test_num_samples": len(manifest["test_clip_keys"]),
    }
    print(f"{action} shared split manifest: {path} (sha256={manifest_sha256[:12]}...)")
    return (
        set(manifest["train_clip_keys"]),
        list(manifest["validation_clip_keys"]),
        list(manifest["test_clip_keys"]),
        meta,
        path,
    )


def resolve_validation_every_n_steps(validation_cfg: dict, train_num_samples: int,
                                     batch_size: int) -> tuple[int, int]:
    explicit_steps = validation_cfg.get("every_n_steps")
    steps_per_epoch = math.ceil(train_num_samples / batch_size)
    if explicit_steps is not None:
        every_n_steps = int(explicit_steps)
    else:
        every_n_epochs = float(validation_cfg.get("every_n_epochs", 0.0))
        if every_n_epochs <= 0:
            raise ValueError("validation requires every_n_steps or a positive every_n_epochs")
        every_n_steps = max(1, math.ceil(steps_per_epoch * every_n_epochs))
    if every_n_steps <= 0:
        raise ValueError(f"validation every_n_steps must be positive, got {every_n_steps}")
    return every_n_steps, steps_per_epoch


def periodic_checkpoint_due(stage_cfg: dict, completed_steps: int) -> bool:
    if stage_cfg.get("checkpoint_on_validation", False):
        validation_cfg = stage_cfg["validation"]
        every_n_steps = validation_cfg.get("every_n_steps")
        return (
            validation_cfg.get("enabled", False)
            and every_n_steps is not None
            and int(every_n_steps) > 0
            and completed_steps % int(every_n_steps) == 0
        )
    every_n_steps = int(stage_cfg["checkpoint_every_n_steps"])
    return every_n_steps > 0 and completed_steps % every_n_steps == 0


def indexed_validation_indices(split_cfg: dict) -> list[int]:
    start = int(split_cfg["validation_every_nth"])
    n_source = int(split_cfg["source_num_samples"])
    one_based = bool(split_cfg.get("one_based_nth", True))
    if one_based:
        return list(range(start - 1, n_source, start))
    return list(range(start, n_source, start))


def build_indexed_split(local_data: str, split_cfg: dict) -> tuple[set[str], list[str], dict]:
    tar_files = find_video_shards(local_data)
    if not tar_files:
        raise FileNotFoundError(f"no video TAR shards found under {local_data}")
    tar_index = int(split_cfg.get("tar_index", 0))
    if tar_index >= len(tar_files):
        raise IndexError(f"requested tar_index={tar_index}, but only found {len(tar_files)} TAR(s)")
    keys = ordered_clip_keys_from_tar(tar_files[tar_index])
    n_source = int(split_cfg["source_num_samples"])
    if len(keys) < n_source:
        raise RuntimeError(f"{tar_files[tar_index]} has {len(keys)} clips, expected at least {n_source}")
    source_keys = keys[:n_source]
    val_indices = indexed_validation_indices(split_cfg)
    val_keys = [source_keys[i] for i in val_indices if 0 <= i < len(source_keys)]
    train_keys = [key for i, key in enumerate(source_keys) if i not in set(val_indices)]
    meta = {
        "tar_path": str(tar_files[tar_index]),
        "tar_index": tar_index,
        "source_num_samples": n_source,
        "train_num_samples": len(train_keys),
        "validation_num_samples": len(val_keys),
        "validation_indices_zero_based": val_indices,
        "validation_indices_one_based": [i + 1 for i in val_indices],
        "one_based_nth": bool(split_cfg.get("one_based_nth", True)),
    }
    return set(train_keys), val_keys, meta


def load_validation_set(local_data: str, val_keys: list[str], cfg: dict,
                        max_tar_files: int = None) -> tuple[torch.Tensor, list[str]]:
    if not val_keys:
        raise RuntimeError("validation key list is empty")
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]
    needed = set(val_keys)
    found = {}
    clip_q, stop_event, reader = iter_clips_parallel(
        local_data, subset_keys=needed, max_tar_files=max_tar_files)
    try:
        while len(found) < len(needed):
            item = clip_q.get(timeout=240)
            if item is None:
                break
            clip_key, mp4_bytes = item
            if clip_key in needed and mp4_bytes:
                found[clip_key] = mp4_bytes
    finally:
        stop_event.set()
        reader.join(timeout=5)
    missing = [key for key in val_keys if key not in found]
    if missing:
        raise RuntimeError(f"missing {len(missing)} validation clips, first missing: {missing[0]}")

    raws = []
    with tempfile.TemporaryDirectory(prefix="stage1_val_decode_") as tmp_dir:
        for key in val_keys:
            decoded = decode_video_bytes(found[key], tmp_dir, key, num_frames)
            if decoded is None:
                raise RuntimeError(f"failed to decode validation clip {key}")
            raws.append(resize_center_crop(decoded, crop_size))
    return torch.stack(raws, dim=0).contiguous(), val_keys


def producer_thread(cfg: dict, q: queue.Queue, stop_event: threading.Event,
                    clip_keys: set, local_data: str, max_tar_files: int = None):
    pcfg = get_pipeline_config()
    batch_size = cfg["optimization"]["batch_size"]
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]
    cfg_aug = cfg["augmentation"]
    decode_workers = pcfg["streaming"]["decode_workers_train"]
    max_retries = pcfg["streaming"]["max_retries"]
    torch.set_num_threads(1)

    tmp_dir = tempfile.mkdtemp(prefix="stage1_jepa_decode_")
    retries = 0

    def _decode_batch(pool, pending_bytes, pending_keys):
        futures = [pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                   for b, k in zip(pending_bytes, pending_keys)]
        results = [(f.result(), k) for f, k in zip(futures, pending_keys)]
        clips = [t for t, _ in results if t is not None]
        keys = [k for t, k in results if t is not None]
        if not clips:
            return
        raw = [augment_clip_consistent(v, cfg_aug, crop_size) for v in clips]
        raw_batch = torch.stack(raw, dim=0).permute(0, 2, 1, 3, 4).contiguous()
        q.put(("batch", raw_batch, keys))

    try:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            while not stop_event.is_set():
                try:
                    clip_q, tar_stop, reader = iter_clips_parallel(
                        local_data,
                        subset_keys=clip_keys or None,
                        max_tar_files=max_tar_files,
                    )
                    pending_bytes, pending_keys = [], []
                    try:
                        while not stop_event.is_set():
                            item = clip_q.get(timeout=120)
                            if item is None:
                                break
                            clip_key, mp4_bytes = item
                            if not mp4_bytes:
                                continue
                            pending_bytes.append(mp4_bytes)
                            pending_keys.append(clip_key)
                            if len(pending_bytes) >= batch_size:
                                _decode_batch(pool, pending_bytes, pending_keys)
                                pending_bytes, pending_keys = [], []
                    finally:
                        tar_stop.set()
                        reader.join(timeout=5)
                    if pending_bytes and not stop_event.is_set():
                        _decode_batch(pool, pending_bytes, pending_keys)
                    if stop_event.is_set():
                        break
                except (ConnectionError, TimeoutError, OSError) as exc:
                    retries += 1
                    if retries > max_retries:
                        print(f"FATAL: data producer failed after {max_retries} retries: {exc}")
                        break
                    wait = min(2 ** retries, 60)
                    print(f"Stream error ({exc}); retry {retries}/{max_retries} in {wait}s")
                    time.sleep(wait)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        q.put(("error" if retries > max_retries else "done", None, None))


def load_cosmos_vae(model_id: str, revision: str, dtype: torch.dtype, device,
                    vae_subfolder: str = "vae", enable_tiling: bool = False,
                    enable_slicing: bool = False) -> CosmosVAEHandle:
    from diffusers import AutoencoderKLWan

    # AutoencoderKLWan is numerically unsafe in reduced precision in several
    # diffusers releases (often producing channel-saturated/green videos).
    # Keep JEPA and the trainable decoder in BF16, but execute the frozen VAE
    # in FP32 as recommended by the official Wan/Cosmos examples.
    vae_dtype = torch.float32
    if dtype != vae_dtype:
        print(f"Using FP32 for Cosmos VAE stability (requested model dtype was {dtype})")
    print(
        f"Loading Cosmos VAE only via HF_ENDPOINT={os.environ.get('HF_ENDPOINT')} "
        f"(subfolder={vae_subfolder})"
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder=vae_subfolder,
        revision=revision,
        torch_dtype=vae_dtype,
        token=HF_TOKEN or None,
    )
    vae.to(device).requires_grad_(False).eval()
    if enable_tiling:
        vae.enable_tiling()
    if enable_slicing:
        vae.enable_slicing()
    latent_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1).float()
    latent_std = torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1).float()
    temporal_scale = 2 ** sum(vae.temperal_downsample)
    print(
        f"Loaded Cosmos VAE only: z_dim={vae.config.z_dim}, "
        f"temporal_scale={temporal_scale}, dtype={vae_dtype}"
    )
    print_cuda_memory("Cosmos VAE load")
    return CosmosVAEHandle(
        vae=vae,
        latent_mean=latent_mean,
        latent_scale=1.0 / latent_std,
        temporal_scale=temporal_scale,
        dtype=vae_dtype,
    )


def normalize_for_cosmos_vae(pipe, latents: torch.Tensor) -> torch.Tensor:
    latent_mean = pipe.latent_mean.to(latents.device, latents.dtype)
    latent_scale = pipe.latent_scale.to(latents.device, latents.dtype)
    return (latents - latent_mean) * latent_scale


def denormalize_from_cosmos_vae(pipe, latents: torch.Tensor) -> torch.Tensor:
    latent_mean = pipe.latent_mean.to(latents.device, latents.dtype)
    latent_scale = pipe.latent_scale.to(latents.device, latents.dtype)
    return latents / latent_scale + latent_mean


def match_num_frames(decoded: torch.Tensor, num_frames: int,
                     temporal_scale: int) -> torch.Tensor:
    if decoded.shape[2] == num_frames:
        return decoded
    decoded = torch.repeat_interleave(decoded, repeats=temporal_scale, dim=2)
    if decoded.shape[2] < num_frames:
        padding = decoded[:, :, -1:].repeat(
            1, 1, num_frames - decoded.shape[2], 1, 1)
        decoded = torch.cat([decoded, padding], dim=2)
    return decoded[:, :, :num_frames]


@torch.no_grad()
def encode_video_latents(pipe, raw_batch: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    del dtype  # Kept in the public signature for existing Stage-1/Stage-2 callers.
    vae_input = raw_batch * 2.0 - 1.0
    with torch.amp.autocast("cuda", enabled=False):
        dist = pipe.vae.encode(vae_input.to(pipe.dtype)).latent_dist
    if hasattr(dist, "mode"):
        latents = dist.mode()
    elif hasattr(dist, "mean"):
        latents = dist.mean
    else:
        latents = dist.sample()
    if not torch.isfinite(latents).all():
        raise RuntimeError("Cosmos VAE encode produced NaN/Inf target latents")
    return normalize_for_cosmos_vae(pipe, latents)


@torch.no_grad()
def decode_latents_to_frames(pipe, latents: torch.Tensor, dtype: torch.dtype, num_frames: int) -> list:
    del dtype  # VAE precision is owned by CosmosVAEHandle, independently of JEPA.
    latents = denormalize_from_cosmos_vae(pipe, latents)
    with torch.amp.autocast("cuda", enabled=False):
        decoded = pipe.vae.decode(latents.to(pipe.dtype), return_dict=False)[0]
    if not torch.isfinite(decoded).all():
        raise RuntimeError("Cosmos VAE decode produced NaN/Inf values")
    decoded = match_num_frames(decoded, num_frames, pipe.temporal_scale)
    decoded = decoded.float().clamp(-1, 1)
    video = ((decoded[0] + 1.0) * 127.5).clamp(0, 255).byte()
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    from PIL import Image
    return [Image.fromarray(frame) for frame in video]


def decode_latents_for_loss(pipe, latents: torch.Tensor, dtype: torch.dtype,
                            num_frames: int) -> torch.Tensor:
    del dtype  # VAE precision is owned by CosmosVAEHandle, independently of JEPA.
    latents = denormalize_from_cosmos_vae(pipe, latents)
    with torch.amp.autocast("cuda", enabled=False):
        decoded = pipe.vae.decode(latents.to(pipe.dtype), return_dict=False)[0]
    if not torch.isfinite(decoded).all():
        raise RuntimeError("Cosmos VAE decode produced NaN/Inf values during loss computation")
    decoded = match_num_frames(decoded, num_frames, pipe.temporal_scale)
    return decoded.float()


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt((pred - target).pow(2) + eps * eps).mean()


def spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor,
                          eps: float) -> torch.Tensor:
    pred_dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    target_dx = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    pred_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    target_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    return charbonnier_loss(pred_dx, target_dx, eps) + charbonnier_loss(pred_dy, target_dy, eps)


def temporal_difference_loss(pred: torch.Tensor, target: torch.Tensor,
                             eps: float) -> torch.Tensor:
    pred_dt = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    target_dt = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    return charbonnier_loss(pred_dt, target_dt, eps)


def flatten_lpips_frames(video: torch.Tensor, max_frames: int) -> torch.Tensor:
    bsz, channels, n_frames, height, width = video.shape
    if channels != 3:
        raise RuntimeError(f"LPIPS expects 3-channel decoded video, got {channels}")
    if max_frames > 0 and n_frames > max_frames:
        frame_idx = torch.linspace(0, n_frames - 1, steps=max_frames, device=video.device).round().long()
        video = video.index_select(2, frame_idx)
        n_frames = max_frames
    return video.permute(0, 2, 1, 3, 4).reshape(bsz * n_frames, channels, height, width)


def build_lpips_model(loss_cfg: dict, device):
    if loss_cfg["lpips_weight"] <= 0:
        return None
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS loss is enabled but the 'lpips' package is not installed. "
            "Install it with: pip install lpips"
        ) from exc
    model = lpips.LPIPS(net=loss_cfg["lpips_net"]).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def compute_stage1_loss(*, pred_latent: torch.Tensor, target_latent: torch.Tensor,
                        raw_batch: torch.Tensor, pipe, dtype: torch.dtype,
                        num_frames: int, loss_cfg: dict, lpips_model,
                        step: int) -> tuple[torch.Tensor, dict]:
    eps = loss_cfg["charbonnier_eps"]
    terms = {}

    latent_l1 = charbonnier_loss(pred_latent.float(), target_latent.detach().float(), eps)
    latent_mse = F.mse_loss(pred_latent.float(), target_latent.detach().float())
    total = (
        loss_cfg["latent_l1_weight"] * latent_l1
        + loss_cfg["latent_mse_weight"] * latent_mse
    )
    terms["latent_l1"] = latent_l1
    terms["latent_mse"] = latent_mse

    decoded_every = int(loss_cfg.get("decoded_loss_every_n_steps", 1))
    run_decoded_losses = decoded_every > 0 and step % decoded_every == 0
    decoded_weight_sum = (
        loss_cfg["rgb_l1_weight"]
        + loss_cfg["spatial_gradient_weight"]
        + loss_cfg["temporal_difference_weight"]
        + loss_cfg["lpips_weight"]
    )

    if run_decoded_losses and decoded_weight_sum > 0:
        pred_video = decode_latents_for_loss(pipe, pred_latent, dtype, num_frames)
        target_video = raw_batch * 2.0 - 1.0
        if pred_video.shape != target_video.shape:
            raise RuntimeError(
                f"decoded video {tuple(pred_video.shape)} != target video {tuple(target_video.shape)}"
            )
        pred_video = pred_video.clamp(-1, 1)
        target_video = target_video.detach().clamp(-1, 1)

        rgb_l1 = charbonnier_loss(pred_video, target_video, eps)
        spatial = spatial_gradient_loss(pred_video, target_video, eps)
        temporal = temporal_difference_loss(pred_video, target_video, eps)
        total = (
            total
            + loss_cfg["rgb_l1_weight"] * rgb_l1
            + loss_cfg["spatial_gradient_weight"] * spatial
            + loss_cfg["temporal_difference_weight"] * temporal
        )
        terms["rgb_l1"] = rgb_l1
        terms["spatial_gradient"] = spatial
        terms["temporal_difference"] = temporal

        if loss_cfg["lpips_weight"] > 0:
            if lpips_model is None:
                raise RuntimeError("LPIPS loss is enabled but lpips_model is None")
            pred_lpips = flatten_lpips_frames(pred_video, loss_cfg["lpips_max_frames"])
            target_lpips = flatten_lpips_frames(target_video, loss_cfg["lpips_max_frames"])
            lpips_loss = lpips_model(pred_lpips.float(), target_lpips.float()).mean()
            total = total + loss_cfg["lpips_weight"] * lpips_loss
            terms["lpips"] = lpips_loss

    for name in ["rgb_l1", "spatial_gradient", "temporal_difference", "lpips"]:
        terms.setdefault(name, pred_latent.new_tensor(0.0))
    terms["total"] = total
    terms["decoded_losses_ran"] = pred_latent.new_tensor(float(run_decoded_losses))
    return total, terms


def raw_batch_to_frames(raw_batch: torch.Tensor) -> list:
    video = (raw_batch[0].detach().float().clamp(0, 1) * 255.0).byte()
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    from PIL import Image
    return [Image.fromarray(frame) for frame in video]


def export_video(frames: list, path: Path, fps: int):
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MP4 export requires imageio and imageio-ffmpeg; the legacy OpenCV "
            "fallback can produce corrupt/green videos. Install with: "
            "python -m pip install imageio imageio-ffmpeg"
        ) from exc
    from diffusers.utils import export_to_video

    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(path), fps=fps)


def append_loss_csv(csv_path: Path, record: dict):
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    columns = ["step", "lr", "decoded_losses_ran"] + LOSS_KEYS
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in columns})


def load_loss_records(log_path: Path) -> list[dict]:
    records = []
    if not log_path.exists():
        return records
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in rec:
                records.append(rec)
    return records


def render_loss_trends(output_dir: Path, log_path: Path):
    records = load_loss_records(log_path)
    if not records:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  WARN: matplotlib unavailable; skipping loss trend plots ({exc})")
        return []

    trend_dir = output_dir / "loss_trends"
    trend_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(r["step"]) for r in records]
    saved = []

    def values_for(key):
        vals = []
        xs = []
        for rec in records:
            val = rec.get(key)
            if val is None:
                continue
            vals.append(float(val))
            xs.append(int(rec["step"]))
        return xs, vals

    plt.figure(figsize=(12, 7))
    for key in LOSS_KEYS:
        xs, vals = values_for(key)
        if vals:
            plt.plot(xs, vals, label=key.replace("loss_", ""))
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Stage-1 Loss Trends")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    combined = trend_dir / "combined_loss_trends.png"
    plt.savefig(combined, dpi=160)
    plt.close()
    saved.append(combined)

    for key in LOSS_KEYS:
        xs, vals = values_for(key)
        if not vals:
            continue
        plt.figure(figsize=(10, 5))
        plt.plot(xs, vals)
        plt.xlabel("step")
        plt.ylabel(key)
        plt.title(key)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        path = trend_dir / f"{key}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        saved.append(path)

    summary = {
        "num_records": len(records),
        "first_step": steps[0],
        "last_step": steps[-1],
        "latest": {key: records[-1].get(key) for key in LOSS_KEYS},
    }
    summary_path = trend_dir / "loss_trends_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    saved.append(summary_path)
    return saved


def hf_upload_enabled(stage_cfg: dict) -> bool:
    return bool(stage_cfg.get("hf_upload", {}).get("enabled", False))


def upload_artifact_to_hf(local_path: Path, output_dir: Path, stage_cfg: dict, artifact_kind: str):
    upload_cfg = stage_cfg.get("hf_upload", {})
    if not upload_cfg.get("enabled", False):
        return
    if not local_path.exists():
        return
    token = HF_TOKEN or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("  WARN: HF upload enabled but HF_TOKEN/HUGGINGFACE_HUB_TOKEN is missing; skipping upload")
        return
    repo_id = upload_cfg.get("repo_id")
    if not repo_id:
        print("  WARN: HF upload enabled but hf_upload.repo_id is empty; skipping upload")
        return
    repo_type = upload_cfg.get("repo_type", "model")
    prefix = upload_cfg.get("path_prefix", "stage1_jepa_decoder").strip("/")
    try:
        rel = local_path.relative_to(output_dir).as_posix()
    except ValueError:
        rel = local_path.name
    path_in_repo = f"{prefix}/{rel}" if prefix else rel
    try:
        from huggingface_hub import HfApi, create_repo

        create_repo(repo_id, repo_type=repo_type, token=token, exist_ok=True)
        HfApi(token=token).upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"stage1: upload {artifact_kind} {rel}",
        )
        print(f"  [hf-upload] {artifact_kind}: {local_path} -> {repo_id}/{path_in_repo}")
    except Exception as exc:
        print(f"  WARN: HF upload failed for {local_path}: {exc}")


def maybe_upload_many(paths: list[Path], output_dir: Path, stage_cfg: dict, artifact_kind: str):
    for path in paths:
        upload_artifact_to_hf(path, output_dir, stage_cfg, artifact_kind)


def init_wandb(stage_cfg: dict, cfg: dict, args, output_dir: Path):
    wandb_cfg = stage_cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    api_key = wandb_cfg.get("api_key") or os.environ.get("WANDB_API_KEY")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging is enabled but wandb is not installed. "
            "Install it with: pip install wandb"
        ) from exc
    if api_key:
        wandb.login(key=api_key, relogin=True)
    public_wandb_cfg = {k: v for k, v in wandb_cfg.items() if k != "api_key"}
    public_args = vars(args).copy()
    public_args.pop("wandb_api_key", None)
    public_args.pop("hf_token", None)
    run_id = getattr(args, "wandb_run_id", None) or os.environ.get("WANDB_RUN_ID") or None
    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg.get("entity") or None,
        id=run_id,
        name=wandb_cfg.get("run_name") or None,
        group=wandb_cfg.get("group") or None,
        tags=wandb_cfg.get("tags") or None,
        dir=str(output_dir),
        config={
            "stage_cfg": {**stage_cfg, "wandb": public_wandb_cfg},
            "model_cfg": cfg.get("model", {}),
            "data_cfg": cfg.get("data", {}),
            "optimization_cfg": cfg.get("optimization", {}),
            "args": public_args,
        },
        resume="must" if run_id else wandb_cfg.get("resume", "allow"),
    )
    print(
        f"W&B logging enabled: project={wandb_cfg['project']} "
        f"run={run.name} id={run.id}"
    )
    return wandb


def wandb_log_metrics(wandb_module, loss_record: dict):
    if wandb_module is None:
        return
    step = int(loss_record["step"])
    metrics = {
        key: loss_record[key]
        for key in ["lr", "decoded_losses_ran"] + LOSS_KEYS
        if key in loss_record
    }
    metrics.update({f"train/{key}": value for key, value in metrics.items()})
    wandb_module.log(metrics, step=step)


def build_validation_summary(step: int, sample_records: list[dict]) -> dict:
    if not sample_records:
        raise ValueError("validation produced no sample records")
    excluded = {"validation_index", "clip_key"}
    metric_keys = sorted({
        key
        for record in sample_records
        for key, value in record.items()
        if key not in excluded and isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    aggregate = {
        key: float(sum(float(record[key]) for record in sample_records if key in record)
                   / sum(1 for record in sample_records if key in record))
        for key in metric_keys
    }
    return {
        "step": int(step),
        "num_samples": len(sample_records),
        "aggregate": aggregate,
        "samples": sample_records,
    }


def wandb_log_validation_summary(wandb_module, summary: dict, step: int, prefix: str):
    if wandb_module is None:
        return
    payload = {
        f"{prefix}/{key}": value
        for key, value in summary["aggregate"].items()
    }
    sample_records = summary["samples"]
    metric_keys = sorted(summary["aggregate"])
    columns = ["validation_index", "clip_key"] + metric_keys
    rows = [[record.get(column) for column in columns] for record in sample_records]
    payload[f"{prefix}/per_sample"] = wandb_module.Table(columns=columns, data=rows)
    wandb_module.log(payload, step=step)


def wandb_log_video(wandb_module, path: Path, step: int, key: str, fps: int):
    if wandb_module is None or not path.exists():
        return
    del fps  # An MP4 path already carries its frame rate; W&B ignores this argument.
    wandb_module.log({key: wandb_module.Video(str(path), format="mp4")}, step=step)


def wandb_log_images(wandb_module, paths: list[Path], step: int, prefix: str):
    if wandb_module is None:
        return
    payload = {}
    for path in paths:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not path.exists():
            continue
        payload[f"{prefix}/{path.stem}"] = wandb_module.Image(str(path))
    if payload:
        wandb_module.log(payload, step=step)


def wandb_save_file(wandb_module, path: Path, output_dir: Path):
    if wandb_module is None or not path.exists():
        return
    try:
        rel = path.relative_to(output_dir)
        wandb_module.save(str(path), base_path=str(output_dir), policy="now")
        print(f"  [wandb] saved file: {rel}")
    except Exception as exc:
        print(f"  WARN: W&B save failed for {path}: {exc}")


def wandb_log_artifact(wandb_module, path: Path, artifact_name: str,
                       artifact_type: str, aliases: list[str] = None):
    if wandb_module is None or not path.exists():
        return
    artifact = wandb_module.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(str(path))
    wandb_module.log_artifact(artifact, aliases=aliases)
    alias_msg = f" aliases={aliases}" if aliases else ""
    print(f"  [wandb] logged artifact: {artifact_name} ({artifact_type}){alias_msg}")


def wandb_log_artifact_many(wandb_module, paths: list[Path], artifact_name: str,
                            artifact_type: str, aliases: list[str] = None):
    if wandb_module is None:
        return
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    artifact = wandb_module.Artifact(artifact_name, type=artifact_type)
    for path in existing:
        artifact.add_file(str(path), name=path.name)
    wandb_module.log_artifact(artifact, aliases=aliases)
    alias_msg = f" aliases={aliases}" if aliases else ""
    print(f"  [wandb] logged artifact: {artifact_name} ({artifact_type}){alias_msg}")


def resolve_source_ckpt(args) -> str:
    if args.source_ckpt and str(args.source_ckpt).lower() not in {"", "none", "null"}:
        path = Path(args.source_ckpt)
        if path.is_dir():
            preferred = [
                "m09c_ckpt_best.pt",
                "m09a_ckpt_best.pt",
                "ckpt_best.pt",
                "checkpoint_best.pt",
                "latest.pt",
                "checkpoint_latest.pt",
                "ckpt_latest.pt",
            ]
            for name in preferred:
                candidate = path / name
                if candidate.exists():
                    print(f"Resolved source checkpoint directory to: {candidate}")
                    return str(candidate)
            matches = []
            for pattern in ("*ckpt*best*.pt", "*best*.pt", "*latest*.pt", "*.pt"):
                matches = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                if matches:
                    print(f"Resolved source checkpoint directory to: {matches[0]}")
                    return str(matches[0])
            raise FileNotFoundError(f"no .pt checkpoint found in source checkpoint directory: {path}")
        return str(path)
    from huggingface_hub import hf_hub_download
    print(f"Downloading source V-JEPA/FactorJEPA checkpoint from HF: "
          f"{args.source_hf_repo}/{args.source_hf_filename}")
    return hf_hub_download(
        repo_id=args.source_hf_repo,
        filename=args.source_hf_filename,
        repo_type=getattr(args, "source_hf_repo_type", DEFAULT_SOURCE_HF_REPO_TYPE),
        token=HF_TOKEN or None,
    )


def load_frozen_jepa(source_ckpt_path: Path, model_cfg: dict, data_cfg: dict, device,
                     dtype: torch.dtype = torch.bfloat16):
    ckpt = torch_load_cpu(source_ckpt_path, weights_only=False)
    if "student" not in ckpt or "predictor" not in ckpt:
        raise KeyError(f"{source_ckpt_path} must contain 'student' and 'predictor' keys")
    student, predictor = build_student_predictor(model_cfg, data_cfg)
    student.load_state_dict(ckpt["student"], strict=False)
    predictor.load_state_dict(ckpt["predictor"], strict=False)
    del ckpt
    gc.collect()
    student = student.to(device=device, dtype=dtype).eval()
    predictor = predictor.to(device=device, dtype=dtype).eval()
    student.requires_grad_(False)
    predictor.requires_grad_(False)
    if hasattr(student, "return_hierarchical"):
        student.return_hierarchical = model_cfg["predict_all"] or model_cfg["n_output_distillation"] > 1
    print(f"Loaded frozen JEPA student + predictor from {source_ckpt_path} ({dtype})")
    print_cuda_memory("JEPA load")
    return student, predictor


def token_grid_shape(model_cfg: dict, data_cfg: dict) -> tuple[int, int, int]:
    t_tokens = data_cfg["num_frames"] // model_cfg["tubelet_size"]
    h_tokens = model_cfg["crop_size"] // model_cfg["patch_size"]
    w_tokens = model_cfg["crop_size"] // model_cfg["patch_size"]
    return t_tokens, h_tokens, w_tokens


def expand_mask(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if isinstance(mask, (list, tuple)):
        if len(mask) != 1:
            raise RuntimeError(f"expected one mask tensor, got {len(mask)}")
        mask = mask[0]
    mask = mask.long()
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    if mask.ndim != 2:
        raise RuntimeError(f"mask must be [B, N] or [N], got {tuple(mask.shape)}")
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    if mask.shape[0] != batch_size:
        raise RuntimeError(f"mask batch {mask.shape[0]} != feature batch {batch_size}")
    return mask


def scatter_features(full_grid: torch.Tensor, token_types: torch.Tensor,
                     indices: torch.Tensor, features: torch.Tensor, token_type: int):
    if indices.shape[1] != features.shape[1]:
        raise RuntimeError(
            f"mask index count {indices.shape[1]} != feature token count {features.shape[1]}"
        )
    bsz, n_total, _ = full_grid.shape
    if int(indices.max()) >= n_total or int(indices.min()) < 0:
        raise RuntimeError(f"mask indices outside full token grid 0..{n_total - 1}")
    batch_idx = torch.arange(bsz, device=full_grid.device).unsqueeze(1).expand_as(indices)
    full_grid[batch_idx, indices] = features
    token_types[batch_idx, indices] = token_type


def select_jepa_width(tokens: torch.Tensor, model_cfg: dict) -> torch.Tensor:
    embed_dim = model_cfg["embed_dim"]
    n_levels = model_cfg["n_output_distillation"]
    if tokens.shape[-1] == embed_dim or tokens.shape[-1] == embed_dim * n_levels:
        return tokens
    raise RuntimeError(
        f"unexpected JEPA token width {tokens.shape[-1]} "
        f"(expected {embed_dim} or {embed_dim * n_levels})"
    )


@torch.no_grad()
def build_full_jepa_grid(raw_batch: torch.Tensor, cfg: dict, mask_generators: list,
                         student, predictor, device,
                         fixed_masks: tuple[torch.Tensor, torch.Tensor] = None
                         ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    bsz = raw_batch.shape[0]
    t_tokens, h_tokens, w_tokens = token_grid_shape(model_cfg, data_cfg)
    n_total = t_tokens * h_tokens * w_tokens
    enc_batch = normalize_for_jepa(raw_batch.to(device))
    mp_cfg = cfg["mixed_precision"]
    jepa_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[mp_cfg["dtype"]]

    with torch.amp.autocast("cuda", dtype=jepa_dtype, enabled=mp_cfg["enabled"]):
        if fixed_masks is None:
            mg = mask_generators[0]
            m_enc, m_pred = mg(bsz)
        else:
            m_enc, m_pred = fixed_masks
        m_enc = expand_mask(m_enc.to(device), bsz)
        m_pred = expand_mask(m_pred.to(device), bsz)
        n_levels = model_cfg["n_output_distillation"]
        z_visible = student(
            enc_batch,
            masks=[m_enc],
            **({"training": True} if n_levels > 1 else {}),
        )
        out = predictor(
            z_visible,
            [m_enc],
            [m_pred],
            **({"mod": "video", "mask_index": 0} if n_levels > 1 else {}),
        )
        if isinstance(out, tuple):
            pred_tokens = out[0]
            visible_tokens = out[1] if out[1] is not None else z_visible
        else:
            pred_tokens = out
            visible_tokens = z_visible

    feature_dtype = jepa_dtype if mp_cfg["enabled"] else torch.float32
    pred_tokens = select_jepa_width(pred_tokens.to(feature_dtype), model_cfg)
    visible_tokens = select_jepa_width(visible_tokens.to(feature_dtype), model_cfg)
    jepa_dim = pred_tokens.shape[-1]
    if visible_tokens.shape[-1] != jepa_dim:
        raise RuntimeError(f"visible width {visible_tokens.shape[-1]} != pred width {jepa_dim}")

    full_grid = torch.zeros(bsz, n_total, jepa_dim, device=device, dtype=feature_dtype)
    token_types = torch.full((bsz, n_total), 2, device=device, dtype=torch.long)
    scatter_features(full_grid, token_types, m_enc, visible_tokens, token_type=0)
    scatter_features(full_grid, token_types, m_pred, pred_tokens, token_type=1)
    meta = {
        "n_total": n_total,
        "n_visible": int(m_enc.shape[1]),
        "n_pred": int(m_pred.shape[1]),
        "jepa_dim": int(jepa_dim),
        "token_grid": [t_tokens, h_tokens, w_tokens],
    }
    return full_grid, token_types, meta


def sample_fixed_validation_masks(mask_generators: list, batch_size: int,
                                  seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one seeded JEPA mask pair for comparable validation videos."""
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            m_enc, m_pred = mask_generators[0](batch_size)
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
    return expand_mask(m_enc, batch_size).cpu(), expand_mask(m_pred, batch_size).cpu()


def validation_seed_for_clip(base_seed: int, clip_key: str) -> int:
    """Stable per-clip seed, independent of Python hash randomization and ordering."""
    digest = hashlib.sha256(f"{int(base_seed)}\0{clip_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2 ** 31)


def build_per_clip_validation_masks(
        mask_generators: list, validation_keys: list[str], base_seed: int,
        shared_masks: tuple[torch.Tensor, torch.Tensor] = None,
        shared_seed: int = None,
        ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[int]]:
    """Build repeatable mask diversity while preserving fixed overfit behavior."""
    masks = []
    seeds = []
    for key in validation_keys:
        if shared_masks is not None:
            masks.append(shared_masks)
            seeds.append(int(base_seed if shared_seed is None else shared_seed))
            continue
        clip_seed = validation_seed_for_clip(base_seed, key)
        masks.append(sample_fixed_validation_masks(mask_generators, 1, clip_seed))
        seeds.append(clip_seed)
    return masks, seeds


class LevelFusion(nn.Module):
    def __init__(self, embed_dim: int, n_levels: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_levels = n_levels
        self.norm = nn.LayerNorm(embed_dim)
        self.level_logits = nn.Parameter(torch.zeros(n_levels))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] == self.embed_dim:
            return self.norm(tokens)
        expected = self.embed_dim * self.n_levels
        if tokens.shape[-1] != expected:
            raise RuntimeError(f"LevelFusion expected width {self.embed_dim} or {expected}, got {tokens.shape[-1]}")
        bsz, n_tokens, _ = tokens.shape
        x = tokens.view(bsz, n_tokens, self.n_levels, self.embed_dim)
        x = self.norm(x)
        weights = torch.softmax(self.level_logits, dim=0).view(1, 1, self.n_levels, 1)
        return (x * weights).sum(dim=2)


class ConvBlock3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size, padding=pad),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Stage1JepaGridDecoder(nn.Module):
    def __init__(self, *, embed_dim: int, n_levels: int, token_grid: tuple[int, int, int],
                 latent_shape: tuple[int, int, int, int], decoder_dim: int,
                 decoder_depth: int, kernel_size: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_levels = n_levels
        self.token_grid = tuple(token_grid)
        self.latent_shape = tuple(latent_shape)
        self.level_fusion = LevelFusion(embed_dim, n_levels)
        self.token_type = nn.Embedding(3, embed_dim)
        self.token_proj = nn.Linear(embed_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, int(np.prod(token_grid)), decoder_dim))
        self.blocks = nn.Sequential(*[
            ConvBlock3D(decoder_dim, kernel_size, dropout)
            for _ in range(decoder_depth)
        ])
        self.out_norm = nn.GroupNorm(8, decoder_dim)
        self.out = nn.Conv3d(decoder_dim, latent_shape[0], kernel_size=1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, full_grid: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, _ = full_grid.shape
        expected_tokens = int(np.prod(self.token_grid))
        if n_tokens != expected_tokens:
            raise RuntimeError(f"decoder expected {expected_tokens} tokens, got {n_tokens}")
        amp_enabled = full_grid.is_cuda and full_grid.dtype in (torch.bfloat16, torch.float16)
        amp_dtype = full_grid.dtype if amp_enabled else torch.bfloat16
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            x = self.level_fusion(full_grid)
            x = x + self.token_type(token_types).to(x.dtype)
            x = self.token_proj(x)
            x = x + self.pos_embed.to(x.dtype)
            t_tokens, h_tokens, w_tokens = self.token_grid
            x = x.transpose(1, 2).reshape(bsz, -1, t_tokens, h_tokens, w_tokens)
            x = self.blocks(x)
            _, target_t, target_h, target_w = self.latent_shape
            x = F.interpolate(
                x, size=(target_t, target_h, target_w), mode="trilinear", align_corners=False)
            return self.out(self.out_norm(x))


def build_decoder(model_cfg: dict, data_cfg: dict, stage_cfg: dict,
                  latent_shape: tuple[int, int, int, int]) -> Stage1JepaGridDecoder:
    return Stage1JepaGridDecoder(
        embed_dim=model_cfg["embed_dim"],
        n_levels=model_cfg["n_output_distillation"],
        token_grid=token_grid_shape(model_cfg, data_cfg),
        latent_shape=latent_shape,
        decoder_dim=stage_cfg["decoder_dim"],
        decoder_depth=stage_cfg["decoder_depth"],
        kernel_size=stage_cfg["decoder_kernel_size"],
        dropout=stage_cfg["dropout"],
    )


def save_checkpoint(path: Path, decoder: Stage1JepaGridDecoder, optimizer, scheduler,
                    step: int, model_cfg: dict, data_cfg: dict, stage_cfg: dict,
                    source_ckpt: str, latent_shape: tuple[int, int, int, int],
                    include_optimizer: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "decoder_state_dict": decoder.state_dict(),
        "step": step,
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
        "stage_cfg": stage_cfg,
        "source_ckpt": source_ckpt,
        "latent_shape": list(latent_shape),
        "token_grid": list(decoder.token_grid),
        "cosmos_model_id": stage_cfg["cosmos_model_id"],
        "cosmos_revision": stage_cfg["cosmos_revision"],
        "has_optimizer": include_optimizer,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_linear_scheduler(optimizer, start_step: int, total_steps: int,
                           warmup_steps: int):
    """Build a fresh schedule or smoothly extend one loaded from a checkpoint."""
    from diffusers.optimization import get_linear_schedule_with_warmup

    if start_step <= 0:
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    if start_step >= total_steps:
        raise ValueError(
            f"resume step {start_step} must be below total training steps {total_steps}"
        )

    # The optimizer checkpoint contains the exact LR used at the resume boundary.
    # Anchor there and decay linearly to zero at the new training horizon, avoiding
    # the LR jump that would result from rebuilding an absolute 0 -> total schedule.
    remaining_steps = total_steps - start_step
    lr_lambdas = []
    for group in optimizer.param_groups:
        current_lr = float(group["lr"])
        base_lr = float(group.get("initial_lr", current_lr))
        if base_lr <= 0:
            base_lr = current_lr
        group["initial_lr"] = base_lr
        anchor_factor = current_lr / base_lr if base_lr > 0 else 1.0

        def lr_lambda(completed_step, *, anchor=start_step, remaining=remaining_steps,
                      factor=anchor_factor):
            progress = max(0, completed_step - anchor)
            return factor * max(0.0, 1.0 - progress / remaining)

        lr_lambdas.append(lr_lambda)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambdas,
        last_epoch=start_step - 1,
    )
    print(
        "Extended linear LR schedule from checkpoint: "
        f"step {start_step} -> {total_steps}, "
        f"starting_lr={[float(group['lr']) for group in optimizer.param_groups]}"
    )
    return scheduler


def step_checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / CHECKPOINT_STEP_TEMPLATE.format(step=step)


@torch.no_grad()
def evaluate_stage1_validation_sample(
        *, step: int, validation_index: int, validation_key: str,
        raw: torch.Tensor, cfg: dict, stage_cfg: dict, pipe, decoder,
        student, predictor, mask_generators: list, dtype: torch.dtype, device,
        fixed_masks: tuple[torch.Tensor, torch.Tensor], validation_mask_seed: int,
        val_dir: Path,
        lpips_model) -> tuple[list[Path], dict]:
    full_grid, token_types, meta = build_full_jepa_grid(
        raw, cfg, mask_generators, student, predictor, device, fixed_masks=fixed_masks)
    pred_latent = decoder(full_grid, token_types)
    target_latent = encode_video_latents(pipe, raw, dtype)
    validation_loss_cfg = dict(stage_cfg["loss"])
    validation_loss_cfg["decoded_loss_every_n_steps"] = 1
    _, loss_terms = compute_stage1_loss(
        pred_latent=pred_latent,
        target_latent=target_latent,
        raw_batch=raw,
        pipe=pipe,
        dtype=dtype,
        num_frames=cfg["data"]["num_frames"],
        loss_cfg=validation_loss_cfg,
        lpips_model=lpips_model,
        step=0,
    )
    record = {
        "validation_index": validation_index,
        "clip_key": validation_key,
        "loss_total": float(loss_terms["total"].item()),
        "loss_latent_l1": float(loss_terms["latent_l1"].item()),
        "loss_latent_mse": float(loss_terms["latent_mse"].item()),
        "loss_rgb_l1": float(loss_terms["rgb_l1"].item()),
        "loss_spatial_gradient": float(loss_terms["spatial_gradient"].item()),
        "loss_temporal_difference": float(loss_terms["temporal_difference"].item()),
        "loss_lpips": float(loss_terms["lpips"].item()),
    }
    prefix = f"val_{validation_index:02d}"
    reconstruction_path = val_dir / f"{prefix}_stage1_reconstruction.mp4"
    roundtrip_path = val_dir / f"{prefix}_cosmos_vae_roundtrip.mp4"
    export_video(
        decode_latents_to_frames(pipe, pred_latent, dtype, cfg["data"]["num_frames"]),
        reconstruction_path,
        stage_cfg["validation"]["fps"],
    )
    export_video(
        decode_latents_to_frames(pipe, target_latent, dtype, cfg["data"]["num_frames"]),
        roundtrip_path,
        stage_cfg["validation"]["fps"],
    )
    json_path = val_dir / f"{prefix}.json"
    json_path.write_text(json.dumps({
        "step": step,
        "validation_index": validation_index,
        "clip_key": validation_key,
        "jepa_grid": meta,
        "pred_latent_shape": list(pred_latent.shape),
        "target_latent_shape": list(target_latent.shape),
        "validation_mask_base_seed": stage_cfg["validation"]["seed"],
        "validation_mask_seed": validation_mask_seed,
        "metrics": record,
    }, indent=2) + "\n")
    return [reconstruction_path, roundtrip_path, json_path], record


@torch.no_grad()
def run_validation(step: int, validation_raw: torch.Tensor, validation_key: str,
                   cfg: dict, stage_cfg: dict, pipe, decoder, student, predictor,
                   mask_generators: list, dtype: torch.dtype, device, output_dir: Path,
                   fixed_masks: tuple[torch.Tensor, torch.Tensor],
                   validation_mask_seed: int, lpips_model):
    decoder.eval()
    val_dir = output_dir / "validation" / f"step_{step:07d}"
    saved, record = evaluate_stage1_validation_sample(
        step=step,
        validation_index=0,
        validation_key=validation_key,
        raw=validation_raw.to(device),
        cfg=cfg,
        stage_cfg=stage_cfg,
        pipe=pipe,
        decoder=decoder,
        student=student,
        predictor=predictor,
        mask_generators=mask_generators,
        dtype=dtype,
        device=device,
        fixed_masks=fixed_masks,
        validation_mask_seed=validation_mask_seed,
        val_dir=val_dir,
        lpips_model=lpips_model,
    )
    summary = build_validation_summary(step, [record])
    metrics_path = val_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    saved.append(metrics_path)
    decoder.train()
    print(f"\n[validation] saved Stage-1 reconstruction and Cosmos VAE roundtrip: {val_dir}")
    return saved, summary


@torch.no_grad()
def run_validation_set(step: int, validation_raw: torch.Tensor, validation_keys: list[str],
                       cfg: dict, stage_cfg: dict, pipe, decoder, student, predictor,
                       mask_generators: list, dtype: torch.dtype, device, output_dir: Path,
                       fixed_masks: list[tuple[torch.Tensor, torch.Tensor]],
                       validation_mask_seeds: list[int], lpips_model):
    if validation_raw.shape[0] == 1:
        key = validation_keys[0] if validation_keys else "unknown"
        return run_validation(
            step, validation_raw, key, cfg, stage_cfg, pipe, decoder, student, predictor,
            mask_generators, dtype, device, output_dir, fixed_masks[0],
            validation_mask_seeds[0], lpips_model)

    decoder.eval()
    val_dir = output_dir / "validation" / f"step_{step:07d}"
    saved = []
    records = []
    for idx, key in enumerate(validation_keys):
        raw = validation_raw[idx:idx + 1].to(device)
        sample_paths, record = evaluate_stage1_validation_sample(
            step=step,
            validation_index=idx,
            validation_key=key,
            raw=raw,
            cfg=cfg,
            stage_cfg=stage_cfg,
            pipe=pipe,
            decoder=decoder,
            student=student,
            predictor=predictor,
            mask_generators=mask_generators,
            dtype=dtype,
            device=device,
            fixed_masks=fixed_masks[idx],
            validation_mask_seed=validation_mask_seeds[idx],
            val_dir=val_dir,
            lpips_model=lpips_model,
        )
        saved.extend(sample_paths)
        records.append(record)
    summary = build_validation_summary(step, records)
    metrics_path = val_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    saved.append(metrics_path)
    decoder.train()
    print(
        f"\n[validation] saved {len(validation_keys)} Stage-1 reconstructions, "
        f"Cosmos VAE roundtrips, and metrics: {val_dir}"
    )
    return saved, summary


def save_validation_references(validation_raw: torch.Tensor, validation_keys: list[str],
                               validation_masks: list[tuple[torch.Tensor, torch.Tensor]],
                               validation_mask_seeds: list[int],
                               cfg: dict, stage_cfg: dict, output_dir: Path):
    val_dir = output_dir / "validation"
    ref_paths = []
    if validation_raw.shape[0] == 1:
        ref_path = val_dir / "reference_input.mp4"
        export_video(raw_batch_to_frames(validation_raw), ref_path, stage_cfg["validation"]["fps"])
        ref_paths.append(ref_path)
    else:
        ref_dir = val_dir / "reference_inputs"
        for idx in range(validation_raw.shape[0]):
            ref_path = ref_dir / f"val_{idx:02d}.mp4"
            export_video(raw_batch_to_frames(validation_raw[idx:idx + 1]), ref_path, stage_cfg["validation"]["fps"])
            ref_paths.append(ref_path)

    manifest_path = val_dir / "reference_input.json"
    manifest_path.write_text(json.dumps({
        "source": "fixed held-out validation set" if validation_raw.shape[0] > 1 else "first successfully decoded training batch",
        "clip_keys": validation_keys,
        "num_validation_samples": len(validation_keys),
        "num_frames": cfg["data"]["num_frames"],
        "crop_size": cfg["data"]["crop_size"],
        "fps": stage_cfg["validation"]["fps"],
        "validation_mask_strategy": "stable_sha256_per_clip",
        "validation_mask_base_seed": stage_cfg["validation"]["seed"],
        "validation_mask_seeds": [
            {"clip_key": key, "seed": seed}
            for key, seed in zip(validation_keys, validation_mask_seeds)
        ],
        "n_visible_tokens": int(validation_masks[0][0].shape[1]),
        "n_pred_tokens": int(validation_masks[0][1].shape[1]),
    }, indent=2) + "\n")
    ref_paths.append(manifest_path)
    return ref_paths


def wandb_log_videos(wandb_module, paths: list[Path], step: int, prefix: str, fps: int):
    for path in paths:
        if path.suffix.lower() != ".mp4":
            continue
        wandb_log_video(wandb_module, path, step, f"{prefix}/{path.stem}", fps)


def train(cfg: dict, stage_cfg: dict, args):
    check_gpu()
    print_cgroup_header(prefix="[stage1-jepa-decoder]")
    start_oom_watchdog(prefix="[stage1-jepa-decoder]-oom-watchdog")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(cfg["data"]["seed"])
    np.random.seed(cfg["data"]["seed"])

    output_dir = Path(args.output_dir)
    wipe_output_dir(output_dir, args.cache_policy, label=f"output_dir ({output_dir.name})")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_ckpt = resolve_source_ckpt(args)
    student, predictor = load_frozen_jepa(
        Path(source_ckpt), cfg["model"], cfg["data"], device, dtype=dtype)
    mask_generators = build_mask_generators(cfg)
    print(f"Mask generators: {len(mask_generators)}")
    pipe = load_cosmos_vae(
        stage_cfg["cosmos_model_id"],
        stage_cfg["cosmos_revision"],
        dtype,
        device,
        vae_subfolder=stage_cfg["cosmos_vae_subfolder"],
        enable_tiling=stage_cfg["cosmos_vae_enable_tiling"],
        enable_slicing=stage_cfg["cosmos_vae_enable_slicing"],
    )
    if stage_cfg["memory"]["vae_gradient_checkpointing"]:
        supports_gc = bool(getattr(pipe.vae, "_supports_gradient_checkpointing", False))
        if supports_gc and hasattr(pipe.vae, "enable_gradient_checkpointing"):
            try:
                pipe.vae.enable_gradient_checkpointing()
                print("Enabled Cosmos VAE gradient checkpointing for decoded Stage-1 losses")
            except (AttributeError, NotImplementedError, ValueError) as exc:
                print(f"  WARN: Cosmos VAE gradient checkpointing unavailable; continuing without it: {exc}")
        else:
            print("  WARN: AutoencoderKLWan does not support gradient checkpointing; "
                  "continuing without it")
    print(f"Loaded frozen Cosmos VAE from {stage_cfg['cosmos_model_id']} ({stage_cfg['cosmos_revision']})")
    lpips_model = build_lpips_model(stage_cfg["loss"], device)
    wandb_module = init_wandb(stage_cfg, cfg, args, output_dir)
    print("Loss weights: "
          f"latent_l1={stage_cfg['loss']['latent_l1_weight']} "
          f"latent_mse={stage_cfg['loss']['latent_mse_weight']} "
          f"rgb_l1={stage_cfg['loss']['rgb_l1_weight']} "
          f"spatial_grad={stage_cfg['loss']['spatial_gradient_weight']} "
          f"temporal_diff={stage_cfg['loss']['temporal_difference_weight']} "
          f"lpips={stage_cfg['loss']['lpips_weight']}")

    subset_keys = load_subset(args.subset) if args.subset else set()
    split_meta = None
    split_validation_raw = None
    split_validation_keys = None
    indexed_split_cfg = stage_cfg.get("indexed_split", {})
    random_split_cfg = stage_cfg.get("random_split", {})
    indexed_split_enabled = bool(indexed_split_cfg.get("enabled", False))
    random_split_enabled = bool(random_split_cfg.get("enabled", False))
    if indexed_split_enabled and random_split_enabled:
        raise ValueError("enable only one of indexed_split or random_split")
    if random_split_enabled and not stage_cfg["overfit"]["enabled"]:
        train_keys, val_keys, test_keys, split_meta, manifest_path = load_or_create_random_split(
            args.local_data,
            random_split_cfg,
            manifest_path=args.split_manifest,
            require_existing=False,
        )
        if subset_keys:
            raise ValueError("--subset cannot be combined with the shared random split")
        subset_keys = train_keys
        split_validation_raw, split_validation_keys = load_validation_set(
            args.local_data, val_keys, cfg, max_tar_files=args.max_tar_files)
        random_split_cfg["manifest_path"] = str(manifest_path)
        random_split_cfg["manifest_sha256"] = split_meta["manifest_sha256"]
        output_manifest_path = output_dir / "dataset_split_manifest.json"
        shutil.copy2(manifest_path, output_manifest_path)
        wandb_save_file(wandb_module, output_manifest_path, output_dir)
        wandb_log_artifact(
            wandb_module,
            output_manifest_path,
            "stage1-dataset-split",
            "dataset-split",
            aliases=["shared-split"],
        )
        print("\n=== Shared deterministic random train/validation/test split ===")
        print(f"TARs: {split_meta['tar_files']}")
        print(f"Training clips: {len(subset_keys)}")
        print(f"Validation clips: {len(split_validation_keys)}")
        print(f"Held-out test clips: {len(test_keys)}")
    elif indexed_split_enabled and not stage_cfg["overfit"]["enabled"]:
        train_keys, val_keys, split_meta = build_indexed_split(args.local_data, indexed_split_cfg)
        if subset_keys:
            train_keys = train_keys & subset_keys
        subset_keys = train_keys
        split_validation_raw, split_validation_keys = load_validation_set(
            args.local_data, val_keys, cfg, max_tar_files=args.max_tar_files)
        split_path = output_dir / "validation_split.json"
        split_path.write_text(json.dumps({
            **split_meta,
            "train_clip_keys": sorted(subset_keys),
            "validation_clip_keys": split_validation_keys,
        }, indent=2) + "\n")
        print("\n=== Fixed indexed train/validation split ===")
        print(f"TAR: {split_meta['tar_path']}")
        print(f"Training clips: {len(subset_keys)}")
        print(f"Validation clips: {len(split_validation_keys)}")
        print(f"Validation one-based indices: {split_meta['validation_indices_one_based']}")

    if random_split_enabled and not stage_cfg["overfit"]["enabled"]:
        val_every, steps_per_epoch = resolve_validation_every_n_steps(
            stage_cfg["validation"], len(subset_keys), cfg["optimization"]["batch_size"])
        stage_cfg["validation"]["every_n_steps"] = val_every
        stage_cfg["validation"]["steps_per_epoch"] = steps_per_epoch
        print(
            f"Epoch schedule: {steps_per_epoch} optimizer steps/epoch; "
            f"validation every {val_every} steps "
            f"({stage_cfg['validation']['every_n_epochs']} epoch)"
        )
        if stage_cfg["checkpoint_on_validation"]:
            print(f"Checkpoint schedule: every validation ({val_every} steps)")

    q = queue.Queue(maxsize=get_pipeline_config()["streaming"]["prefetch_queue_train"])
    stop_event = threading.Event()
    prod = threading.Thread(
        target=producer_thread,
        args=(cfg, q, stop_event, subset_keys, args.local_data, args.max_tar_files),
        daemon=True,
    )
    prod.start()

    total_steps = stage_cfg["num_training_steps"]
    decoder = optimizer = scheduler = None
    ckpt_path = output_dir / CHECKPOINT_LATEST
    resume_payload = torch_load_cpu(ckpt_path, weights_only=False) if ckpt_path.exists() else None
    if resume_payload is not None and random_split_enabled and not stage_cfg["overfit"]["enabled"]:
        resume_manifest_hash = (
            resume_payload.get("stage_cfg", {})
            .get("random_split", {})
            .get("manifest_sha256")
        )
        if resume_manifest_hash != split_meta["manifest_sha256"]:
            raise RuntimeError(
                "latest Stage-1 checkpoint was trained with a different split manifest: "
                f"checkpoint={resume_manifest_hash}, current={split_meta['manifest_sha256']}"
            )
    start_step = int(resume_payload["step"]) if resume_payload is not None else 0

    pbar = make_pbar(total=total_steps, initial=start_step, desc="stage1_jepa_grid_decoder", unit="step")
    log_path = output_dir / "loss_log.jsonl"
    csv_path = output_dir / "loss_log.csv"
    log_file = log_path.open("a")
    validation_raw = None
    validation_keys = None
    validation_masks = None
    validation_mask_seeds = None
    overfit_raw_batch = None
    overfit_batch_keys = None
    overfit_train_masks = None
    ran_step0_validation = start_step > 0

    print(f"\n=== Stage-1 JEPA-grid -> Cosmos-VAE-latent decoder: {start_step} -> {total_steps} steps ===")
    print("Frozen: JEPA student, JEPA predictor, Cosmos VAE")
    print("Trainable: level fusion, token-grid decoder, latent head")
    if start_step > 0 and stage_cfg["validation"]["run_before_training"]:
        print("Resume mode: skipping the step-0 validation already recorded by the original run")
    if stage_cfg["overfit"]["enabled"]:
        print("OVERFIT MODE: training on exactly one cached sample, fixed JEPA mask, latent MSE only")

    try:
        for step in range(start_step, total_steps):
            if stage_cfg["overfit"]["enabled"] and overfit_raw_batch is not None:
                raw_batch = overfit_raw_batch.to(device)
                batch_keys = overfit_batch_keys
                fixed_train_masks = overfit_train_masks
            else:
                try:
                    msg_type, raw_batch, batch_keys = q.get(timeout=600)
                except queue.Empty:
                    print(f"Producer timeout at step {step}; stopping.")
                    break
                if msg_type == "error":
                    raise RuntimeError("producer failed")
                if msg_type == "done":
                    print(f"\nData exhausted at step {step}/{total_steps}")
                    break
                if stage_cfg["overfit"]["enabled"]:
                    raw_batch = raw_batch[:1].contiguous()
                    batch_keys = [batch_keys[0] if batch_keys else "unknown"]
                    overfit_raw_batch = raw_batch.detach().cpu()
                    overfit_batch_keys = batch_keys
                    overfit_train_masks = sample_fixed_validation_masks(
                        mask_generators,
                        batch_size=1,
                        seed=stage_cfg["overfit"]["fixed_mask_seed"],
                    )
                    fixed_train_masks = overfit_train_masks
                    stop_event.set()
                    print(f"\n[overfit] fixed training clip: {batch_keys[0]}")
                    print(f"[overfit] fixed train JEPA mask seed: {stage_cfg['overfit']['fixed_mask_seed']}")
                else:
                    fixed_train_masks = None
                raw_batch = raw_batch.to(device)
            with torch.no_grad():
                full_grid, token_types, grid_meta = build_full_jepa_grid(
                    raw_batch, cfg, mask_generators, student, predictor, device,
                    fixed_masks=fixed_train_masks)
                target_latent = encode_video_latents(pipe, raw_batch, dtype)

            latent_shape = tuple(target_latent.shape[1:])
            if decoder is None:
                decoder = build_decoder(cfg["model"], cfg["data"], stage_cfg, latent_shape).to(device)
                optimizer = torch.optim.AdamW(
                    decoder.parameters(),
                    lr=stage_cfg["learning_rate"],
                    weight_decay=stage_cfg["weight_decay"],
                )
                if resume_payload is not None:
                    if tuple(resume_payload["latent_shape"]) != latent_shape:
                        raise RuntimeError(
                            f"resume latent shape {resume_payload['latent_shape']} != current {latent_shape}"
                        )
                    decoder.load_state_dict(resume_payload["decoder_state_dict"])
                    if "optimizer" in resume_payload:
                        optimizer.load_state_dict(resume_payload["optimizer"])
                    else:
                        print("  WARN: resume checkpoint has no optimizer state; using a fresh optimizer")
                    scheduler = build_linear_scheduler(
                        optimizer,
                        start_step=start_step,
                        total_steps=total_steps,
                        warmup_steps=stage_cfg["scheduler_warmup_steps"],
                    )
                    del resume_payload
                    resume_payload = None
                    gc.collect()
                    print(f"Resumed Stage-1 decoder from step {start_step}")
                else:
                    scheduler = build_linear_scheduler(
                        optimizer,
                        start_step=0,
                        total_steps=total_steps,
                        warmup_steps=stage_cfg["scheduler_warmup_steps"],
                    )
                n_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
                print(f"Decoder token grid: {decoder.token_grid}; target latent shape: {latent_shape}")
                print(f"Trainable decoder params: {n_trainable / 1e6:.1f}M")
                print_cuda_memory("Stage-1 decoder initialization")

            if validation_raw is None and stage_cfg["validation"]["enabled"]:
                if split_validation_raw is not None:
                    validation_raw = split_validation_raw.detach().cpu()
                    validation_keys = split_validation_keys
                else:
                    validation_raw = raw_batch[:1].detach().cpu()
                    validation_keys = [batch_keys[0] if batch_keys else "unknown"]
                if stage_cfg["overfit"]["enabled"]:
                    validation_masks, validation_mask_seeds = build_per_clip_validation_masks(
                        mask_generators,
                        validation_keys,
                        stage_cfg["validation"]["seed"],
                        shared_masks=overfit_train_masks,
                        shared_seed=stage_cfg["overfit"]["fixed_mask_seed"],
                    )
                else:
                    validation_masks, validation_mask_seeds = build_per_clip_validation_masks(
                        mask_generators,
                        validation_keys,
                        stage_cfg["validation"]["seed"],
                    )
                ref_paths = save_validation_references(
                    validation_raw, validation_keys, validation_masks,
                    validation_mask_seeds, cfg, stage_cfg, output_dir)
                print(f"\n[validation] fixed reference clips: {len(validation_keys)}")
                if stage_cfg["overfit"]["enabled"]:
                    print(
                        f"[validation] shared overfit JEPA mask seed: "
                        f"{stage_cfg['overfit']['fixed_mask_seed']}"
                    )
                else:
                    print(
                        f"[validation] deterministic per-clip JEPA masks: "
                        f"base seed={stage_cfg['validation']['seed']}"
                    )
                print(f"[validation] saved reference videos under: {output_dir / 'validation'}")
                maybe_upload_many(ref_paths, output_dir, stage_cfg, "validation-reference")
                wandb_log_videos(
                    wandb_module, ref_paths, start_step,
                    "validation/reference_input", stage_cfg["validation"]["fps"])
                wandb_log_artifact_many(
                    wandb_module, ref_paths,
                    "stage1-validation-reference", "validation-reference")

            if (stage_cfg["validation"]["enabled"] and stage_cfg["validation"]["run_before_training"]
                    and not ran_step0_validation and step == start_step):
                val_paths, val_summary = run_validation_set(
                    0, validation_raw, validation_keys, cfg, stage_cfg, pipe, decoder,
                    student, predictor, mask_generators, dtype, device, output_dir,
                    fixed_masks=validation_masks,
                    validation_mask_seeds=validation_mask_seeds,
                    lpips_model=lpips_model)
                maybe_upload_many(val_paths, output_dir, stage_cfg, "validation")
                if val_paths:
                    wandb_log_videos(
                        wandb_module, val_paths, 0,
                        "validation/stage1/videos", stage_cfg["validation"]["fps"])
                    wandb_log_validation_summary(
                        wandb_module, val_summary, 0, "validation/stage1")
                    wandb_log_artifact_many(
                        wandb_module, val_paths,
                        "stage1-validation-step-0000000", "validation")
                ran_step0_validation = True
                cuda_cleanup()

            decoder.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                pred_latent = decoder(full_grid.detach(), token_types)
                if pred_latent.shape != target_latent.shape:
                    raise RuntimeError(
                        f"decoder output {tuple(pred_latent.shape)} != target latent {tuple(target_latent.shape)}"
                    )
                loss, loss_terms = compute_stage1_loss(
                    pred_latent=pred_latent,
                    target_latent=target_latent,
                    raw_batch=raw_batch,
                    pipe=pipe,
                    dtype=dtype,
                    num_frames=cfg["data"]["num_frames"],
                    loss_cfg=stage_cfg["loss"],
                    lpips_model=lpips_model,
                    step=step,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), stage_cfg["grad_clip"])
                optimizer.step()
                scheduler.step()

            lr_val = scheduler.get_last_lr()[0]
            loss_record = {
                "step": step,
                "loss_total": round(float(loss_terms["total"].item()), 8),
                "loss_latent_l1": round(float(loss_terms["latent_l1"].item()), 8),
                "loss_latent_mse": round(float(loss_terms["latent_mse"].item()), 8),
                "loss_rgb_l1": round(float(loss_terms["rgb_l1"].item()), 8),
                "loss_spatial_gradient": round(float(loss_terms["spatial_gradient"].item()), 8),
                "loss_temporal_difference": round(float(loss_terms["temporal_difference"].item()), 8),
                "loss_lpips": round(float(loss_terms["lpips"].item()), 8),
                "decoded_losses_ran": bool(loss_terms["decoded_losses_ran"].item()),
                "lr": lr_val,
                "jepa_grid": grid_meta,
                "target_latent_shape": list(target_latent.shape),
            }
            log_file.write(json.dumps(loss_record) + "\n")
            log_file.flush()
            os.fsync(log_file.fileno())
            append_loss_csv(csv_path, loss_record)
            wandb_log_metrics(wandb_module, loss_record)
            pbar.set_postfix_str(
                f"loss={loss.item():.5f} latent={loss_terms['latent_mse'].item():.5f} lr={lr_val:.2e}"
            )
            pbar.update(1)

            if periodic_checkpoint_due(stage_cfg, step + 1):
                step_ckpt_path = step_checkpoint_path(output_dir, step + 1)
                save_checkpoint(
                    step_ckpt_path, decoder, optimizer, scheduler, step + 1,
                    cfg["model"], cfg["data"], stage_cfg, source_ckpt, latent_shape,
                    include_optimizer=stage_cfg["checkpoint_include_optimizer_for_step_files"])
                save_checkpoint(
                    ckpt_path, decoder, optimizer, scheduler, step + 1,
                    cfg["model"], cfg["data"], stage_cfg, source_ckpt, latent_shape,
                    include_optimizer=True)
                upload_artifact_to_hf(step_ckpt_path, output_dir, stage_cfg, "checkpoint")
                upload_artifact_to_hf(ckpt_path, output_dir, stage_cfg, "checkpoint-latest")
                wandb_save_file(wandb_module, step_ckpt_path, output_dir)
                wandb_save_file(wandb_module, ckpt_path, output_dir)
                wandb_log_artifact(
                    wandb_module, step_ckpt_path,
                    f"stage1-decoder-step-{step + 1:07d}", "model",
                    aliases=[f"step-{step + 1:07d}"])
                wandb_log_artifact(
                    wandb_module, ckpt_path,
                    "stage1-decoder-latest", "model",
                    aliases=["latest"])
            trend_every = int(stage_cfg["loss_trends"]["every_n_steps"])
            if trend_every > 0 and (step + 1) % trend_every == 0:
                trend_paths = render_loss_trends(output_dir, log_path)
                maybe_upload_many(trend_paths + [log_path, csv_path], output_dir, stage_cfg, "loss-trends")
                wandb_log_images(wandb_module, trend_paths, step + 1, "loss_trends")
                wandb_log_artifact_many(
                    wandb_module, trend_paths + [log_path, csv_path],
                    f"stage1-loss-trends-step-{step + 1:07d}", "loss-trends")
            val_every = stage_cfg["validation"]["every_n_steps"]
            if (stage_cfg["validation"]["enabled"] and validation_raw is not None
                    and val_every > 0 and (step + 1) % val_every == 0):
                val_paths, val_summary = run_validation_set(
                    step + 1, validation_raw, validation_keys, cfg, stage_cfg, pipe, decoder,
                    student, predictor, mask_generators, dtype, device, output_dir,
                    fixed_masks=validation_masks,
                    validation_mask_seeds=validation_mask_seeds,
                    lpips_model=lpips_model)
                maybe_upload_many(val_paths, output_dir, stage_cfg, "validation")
                if val_paths:
                    wandb_log_videos(
                        wandb_module, val_paths, step + 1,
                        "validation/stage1/videos", stage_cfg["validation"]["fps"])
                    wandb_log_validation_summary(
                        wandb_module, val_summary, step + 1, "validation/stage1")
                    wandb_log_artifact_many(
                        wandb_module, val_paths,
                        f"stage1-validation-step-{step + 1:07d}", "validation")
                cuda_cleanup()

    except KeyboardInterrupt:
        print("\nInterrupted; saving latest checkpoint.")
    finally:
        pbar.close()
        log_file.close()
        stop_event.set()
        if decoder is not None:
            last_step_ckpt_path = step_checkpoint_path(output_dir, step + 1)
            if not last_step_ckpt_path.exists():
                save_checkpoint(
                    last_step_ckpt_path, decoder, optimizer, scheduler, step + 1,
                    cfg["model"], cfg["data"], stage_cfg, source_ckpt, tuple(decoder.latent_shape),
                    include_optimizer=stage_cfg["checkpoint_include_optimizer_for_step_files"])
            save_checkpoint(
                ckpt_path, decoder, optimizer, scheduler, step + 1,
                cfg["model"], cfg["data"], stage_cfg, source_ckpt, tuple(decoder.latent_shape),
                include_optimizer=True)

    if decoder is None:
        raise RuntimeError("0 batches reached decoder initialization")
    final_path = output_dir / CHECKPOINT_FINAL
    save_checkpoint(
        final_path, decoder, optimizer, scheduler, step + 1,
        cfg["model"], cfg["data"], stage_cfg, source_ckpt, tuple(decoder.latent_shape),
        include_optimizer=stage_cfg["checkpoint_include_optimizer_for_final"])
    final_step_path = step_checkpoint_path(output_dir, step + 1)
    if not final_step_path.exists():
        save_checkpoint(
            final_step_path, decoder, optimizer, scheduler, step + 1,
            cfg["model"], cfg["data"], stage_cfg, source_ckpt, tuple(decoder.latent_shape),
            include_optimizer=stage_cfg["checkpoint_include_optimizer_for_step_files"])
    trend_paths = render_loss_trends(output_dir, log_path)
    maybe_upload_many(trend_paths + [log_path, csv_path, final_path, ckpt_path, final_step_path], output_dir, stage_cfg, "final")
    wandb_log_images(wandb_module, trend_paths, step + 1, "loss_trends")
    wandb_log_artifact_many(
        wandb_module, trend_paths + [log_path, csv_path],
        "stage1-loss-trends-final", "loss-trends")
    wandb_save_file(wandb_module, final_path, output_dir)
    wandb_save_file(wandb_module, final_step_path, output_dir)
    wandb_save_file(wandb_module, ckpt_path, output_dir)
    wandb_log_artifact(wandb_module, final_path, "stage1-decoder-final", "model", aliases=["final"])
    wandb_log_artifact(
        wandb_module, final_step_path,
        f"stage1-decoder-step-{step + 1:07d}", "model",
        aliases=[f"step-{step + 1:07d}"])
    wandb_log_artifact(wandb_module, ckpt_path, "stage1-decoder-latest", "model", aliases=["latest"])
    if wandb_module is not None:
        wandb_module.finish()
    print(f"\nSaved Stage-1 decoder: {final_path}")


def main():
    parser = argparse.ArgumentParser("Train Stage-1 JEPA-grid -> Cosmos VAE latent decoder")
    parser.add_argument("--SANITY", action="store_true")
    parser.add_argument("--POC", action="store_true")
    parser.add_argument("--FULL", action="store_true")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--stage1-config", default="configs/stage1_jepa_decoder.yaml")
    parser.add_argument("--source-ckpt", default=DEFAULT_SOURCE_CKPT)
    parser.add_argument("--source-hf-repo", default=DEFAULT_SOURCE_HF_REPO)
    parser.add_argument("--source-hf-repo-type", default=DEFAULT_SOURCE_HF_REPO_TYPE)
    parser.add_argument("--source-hf-filename", default=DEFAULT_SOURCE_HF_FILENAME)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--local-data", default=None)
    parser.add_argument(
        "--dataset-id",
        default=DENSEWORLD_DATASET_ID,
        help="HF dataset repository used only when the requested local TARs are missing.",
    )
    parser.add_argument("--max-tar-files", type=int, default=None)
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Shared random split manifest. Stage-1 creates it once and reuses it on resume.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--disable-validation", action="store_true")
    parser.add_argument("--validation-every-n-steps", type=int, default=None)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--loss-trends-every-n-steps", type=int, default=None)
    parser.add_argument("--overfit-one-sample", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=None)
    parser.add_argument("--overfit-mask-seed", type=int, default=None)
    parser.add_argument("--hf-upload", action="store_true")
    parser.add_argument("--hf-upload-repo", default=None)
    parser.add_argument("--hf-upload-repo-type", default=None)
    parser.add_argument("--hf-upload-path-prefix", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-api-key", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="Existing W&B run ID to resume. Uses resume='must' so an invalid ID fails loudly.",
    )
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--hf-token", default=None)
    add_cache_policy_arg(parser)
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        global HF_TOKEN
        HF_TOKEN = args.hf_token
    args.hf_dataset_repo = args.dataset_id
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)
    if not (args.SANITY or args.POC or args.FULL):
        raise SystemExit("Specify --SANITY, --POC, or --FULL")

    cfg = load_merged_config(args.model_config, args.train_config)
    stage_cfg = load_config(args.stage1_config)
    stage_cfg.setdefault("cosmos_vae_subfolder", "vae")
    stage_cfg.setdefault("cosmos_vae_enable_tiling", False)
    stage_cfg.setdefault("cosmos_vae_enable_slicing", False)
    stage_cfg.setdefault("checkpoint_on_validation", False)
    stage_cfg.setdefault("checkpoint_include_optimizer_for_step_files", False)
    stage_cfg.setdefault("checkpoint_include_optimizer_for_final", False)
    stage_cfg.setdefault("memory", {})
    stage_cfg["memory"].setdefault("vae_gradient_checkpointing", True)
    stage_cfg.setdefault("loss", {})
    stage_cfg["loss"].setdefault("latent_l1_weight", 0.1)
    stage_cfg["loss"].setdefault("latent_mse_weight", 1.0)
    stage_cfg["loss"].setdefault("rgb_l1_weight", 0.2)
    stage_cfg["loss"].setdefault("spatial_gradient_weight", 0.05)
    stage_cfg["loss"].setdefault("temporal_difference_weight", 0.05)
    stage_cfg["loss"].setdefault("lpips_weight", 0.05)
    stage_cfg["loss"].setdefault("lpips_net", "alex")
    stage_cfg["loss"].setdefault("lpips_max_frames", 4)
    stage_cfg["loss"].setdefault("charbonnier_eps", 1.0e-3)
    stage_cfg["loss"].setdefault("decoded_loss_every_n_steps", 1)
    stage_cfg.setdefault("overfit", {})
    stage_cfg["overfit"].setdefault("enabled", False)
    stage_cfg["overfit"].setdefault("num_training_steps", 1000)
    stage_cfg["overfit"].setdefault("fixed_mask_seed", 1)
    stage_cfg["overfit"].setdefault("latent_mse_only", True)
    stage_cfg["overfit"].setdefault("scheduler_warmup_steps", 0)
    stage_cfg.setdefault("indexed_split", {})
    stage_cfg["indexed_split"].setdefault("enabled", False)
    stage_cfg["indexed_split"].setdefault("tar_index", 0)
    stage_cfg["indexed_split"].setdefault("source_num_samples", 1000)
    stage_cfg["indexed_split"].setdefault("validation_every_nth", 50)
    stage_cfg["indexed_split"].setdefault("one_based_nth", True)
    stage_cfg["indexed_split"].setdefault("num_training_steps", 14000)
    stage_cfg["indexed_split"].setdefault("max_tar_files", stage_cfg["indexed_split"]["tar_index"] + 1)
    stage_cfg.setdefault("random_split", {})
    stage_cfg["random_split"].setdefault("enabled", False)
    stage_cfg["random_split"].setdefault("manifest_path", "data/splits/stage12_split.json")
    stage_cfg["random_split"].setdefault("seed", 42)
    stage_cfg["random_split"].setdefault("max_tar_files", 3)
    stage_cfg["random_split"].setdefault("source_num_samples", 3000)
    stage_cfg["random_split"].setdefault("test_fraction", 0.10)
    stage_cfg["random_split"].setdefault("validation_num_samples", 50)
    stage_cfg["random_split"].setdefault("num_training_steps", 14000)
    stage_cfg.setdefault("loss_trends", {})
    stage_cfg["loss_trends"].setdefault("every_n_steps", stage_cfg["checkpoint_every_n_steps"])
    stage_cfg.setdefault("hf_upload", {})
    stage_cfg["hf_upload"].setdefault("enabled", False)
    stage_cfg["hf_upload"].setdefault("repo_id", "")
    stage_cfg["hf_upload"].setdefault("repo_type", "model")
    stage_cfg["hf_upload"].setdefault("path_prefix", "stage1_jepa_decoder")
    stage_cfg.setdefault("wandb", {})
    stage_cfg["wandb"].setdefault("enabled", False)
    stage_cfg["wandb"].setdefault("project", "factorjepa-stage1")
    stage_cfg["wandb"].setdefault("entity", "")
    stage_cfg["wandb"].setdefault("run_name", "")
    stage_cfg["wandb"].setdefault("group", "")
    stage_cfg["wandb"].setdefault("tags", ["stage1", "jepa-decoder", "cosmos-vae"])
    stage_cfg["wandb"].setdefault("resume", "allow")
    stage_cfg["wandb"].setdefault("api_key", "")
    stage_cfg.setdefault("validation", {})
    stage_cfg["validation"].setdefault("enabled", True)
    stage_cfg["validation"].setdefault("run_before_training", True)
    stage_cfg["validation"].setdefault("every_n_steps", stage_cfg["checkpoint_every_n_steps"])
    stage_cfg["validation"].setdefault("every_n_epochs", None)
    stage_cfg["validation"].setdefault("fps", 16)
    stage_cfg["validation"].setdefault("seed", 1)
    if args.disable_validation:
        stage_cfg["validation"]["enabled"] = False
    if args.validation_every_n_steps is not None:
        stage_cfg["validation"]["every_n_steps"] = args.validation_every_n_steps
    if args.validation_seed is not None:
        stage_cfg["validation"]["seed"] = args.validation_seed
    if args.loss_trends_every_n_steps is not None:
        stage_cfg["loss_trends"]["every_n_steps"] = args.loss_trends_every_n_steps
    if args.overfit_one_sample:
        stage_cfg["overfit"]["enabled"] = True
    if args.overfit_steps is not None:
        stage_cfg["overfit"]["num_training_steps"] = args.overfit_steps
    if args.overfit_mask_seed is not None:
        stage_cfg["overfit"]["fixed_mask_seed"] = args.overfit_mask_seed
    if args.hf_upload:
        stage_cfg["hf_upload"]["enabled"] = True
    if args.hf_upload_repo is not None:
        stage_cfg["hf_upload"]["repo_id"] = args.hf_upload_repo
    if args.hf_upload_repo_type is not None:
        stage_cfg["hf_upload"]["repo_type"] = args.hf_upload_repo_type
    if args.hf_upload_path_prefix is not None:
        stage_cfg["hf_upload"]["path_prefix"] = args.hf_upload_path_prefix
    if args.wandb:
        stage_cfg["wandb"]["enabled"] = True
    if args.wandb_api_key is not None:
        stage_cfg["wandb"]["api_key"] = args.wandb_api_key
    if args.wandb_project is not None:
        stage_cfg["wandb"]["project"] = args.wandb_project
    if args.wandb_entity is not None:
        stage_cfg["wandb"]["entity"] = args.wandb_entity
    if args.wandb_run_name is not None:
        stage_cfg["wandb"]["run_name"] = args.wandb_run_name
    if args.wandb_group is not None:
        stage_cfg["wandb"]["group"] = args.wandb_group
    if args.batch_size is not None:
        cfg["optimization"]["batch_size"] = args.batch_size
    if stage_cfg["overfit"]["enabled"]:
        cfg["optimization"]["batch_size"] = 1
        if args.max_tar_files is None:
            args.max_tar_files = 1
        stage_cfg["scheduler_warmup_steps"] = stage_cfg["overfit"]["scheduler_warmup_steps"]
        if stage_cfg["overfit"]["latent_mse_only"]:
            stage_cfg["loss"]["latent_l1_weight"] = 0.0
            stage_cfg["loss"]["latent_mse_weight"] = 1.0
            stage_cfg["loss"]["rgb_l1_weight"] = 0.0
            stage_cfg["loss"]["spatial_gradient_weight"] = 0.0
            stage_cfg["loss"]["temporal_difference_weight"] = 0.0
            stage_cfg["loss"]["lpips_weight"] = 0.0
            stage_cfg["loss"]["decoded_loss_every_n_steps"] = 0
    split_for_data = (
        stage_cfg["random_split"]
        if stage_cfg["random_split"]["enabled"] else stage_cfg["indexed_split"]
    )
    if stage_cfg["random_split"]["enabled"] and not stage_cfg["overfit"]["enabled"]:
        expected_max_tars = int(stage_cfg["random_split"]["max_tar_files"])
        if args.max_tar_files is not None and args.max_tar_files != expected_max_tars:
            raise ValueError(
                f"--max-tar-files={args.max_tar_files} conflicts with shared random split "
                f"max_tar_files={expected_max_tars}"
            )
        args.max_tar_files = expected_max_tars
    if split_for_data["enabled"] and not stage_cfg["overfit"]["enabled"]:
        if args.max_tar_files is None:
            args.max_tar_files = split_for_data["max_tar_files"]
    args.local_data = ensure_local_data(args)
    mode_key = "sanity" if args.SANITY else ("poc" if args.POC else "full")
    stage_cfg["num_training_steps"] = stage_cfg["num_training_steps_by_mode"][mode_key]
    if stage_cfg["overfit"]["enabled"]:
        stage_cfg["num_training_steps"] = stage_cfg["overfit"]["num_training_steps"]
    elif stage_cfg["random_split"]["enabled"]:
        stage_cfg["num_training_steps"] = stage_cfg["random_split"]["num_training_steps"]
    elif stage_cfg["indexed_split"]["enabled"]:
        stage_cfg["num_training_steps"] = stage_cfg["indexed_split"]["num_training_steps"]
    if stage_cfg["validation"]["every_n_steps"] is None and (
            stage_cfg["overfit"]["enabled"] or not stage_cfg["random_split"]["enabled"]):
        stage_cfg["validation"]["every_n_steps"] = stage_cfg["checkpoint_every_n_steps"]
    train(cfg, stage_cfg, args)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback
        print(f"\nFATAL (stage1-jepa-decoder): {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
