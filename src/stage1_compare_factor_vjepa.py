"""Compare one trained Stage-1 decoder on FactorJEPA and vanilla V-JEPA 2.1.

This is intentionally a decoder-transfer comparison: both frozen JEPA models use
the same input frames and per-clip mask, then the same FactorJEPA-trained Stage-1
decoder maps their combined visible/predicted token grids into Cosmos VAE space.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_ENDPOINT"] = os.environ.get("FACTORJEPA_HF_ENDPOINT", "https://huggingface.co")
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = os.environ["HF_ENDPOINT"]

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import stage1_jepa_decoder_train as stage1_train
from stage1_jepa_decoder_train import (
    DEFAULT_SOURCE_HF_FILENAME,
    DEFAULT_SOURCE_HF_REPO,
    DEFAULT_SOURCE_HF_REPO_TYPE,
    Stage1JepaGridDecoder,
    build_full_jepa_grid,
    build_per_clip_validation_masks,
    denormalize_from_cosmos_vae,
    encode_video_latents,
    export_video,
    flatten_lpips_frames,
    load_cosmos_vae,
    resize_center_crop,
    torch_load_cpu,
    validation_seed_for_clip,
)
from utils.config import check_gpu, load_merged_config
from utils.gpu_batch import cuda_cleanup
from utils.training import build_mask_generators, build_student_predictor
from utils.video_io import decode_video_bytes


DEFAULT_VJEPA_CKPT = "checkpoints/vjepa2_1_vitg_384.pt"
DEFAULT_VJEPA_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt"
CONTRACT_KEYS = (
    "arch", "embed_dim", "pred_embed_dim", "pred_depth", "pred_num_heads",
    "n_output_distillation", "crop_size", "patch_size", "tubelet_size", "predict_all",
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Compare FactorJEPA and vanilla V-JEPA with one trained Stage-1 decoder")
    parser.add_argument("--clips-dir", required=True, help="Directory searched recursively for MP4s")
    parser.add_argument("--decoder-ckpt", required=True, help="Trained Stage-1 decoder checkpoint")
    parser.add_argument(
        "--factorjepa-ckpt", default=None,
        help="Local FactorJEPA student+predictor checkpoint")
    parser.add_argument("--factorjepa-hf-repo", default=DEFAULT_SOURCE_HF_REPO)
    parser.add_argument("--factorjepa-hf-repo-type", default=DEFAULT_SOURCE_HF_REPO_TYPE)
    parser.add_argument("--factorjepa-hf-filename", default=DEFAULT_SOURCE_HF_FILENAME)
    parser.add_argument("--vjepa-ckpt", default=DEFAULT_VJEPA_CKPT)
    parser.add_argument("--vjepa-url", default=DEFAULT_VJEPA_URL)
    parser.add_argument("--model-config", default="configs/model/vjepa2_1_vitg.yaml")
    parser.add_argument(
        "--train-config", default="configs/train/surgery_3stage_DI_diheavy_encoder.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mask-seed", type=int, default=1)
    parser.add_argument(
        "--mask-ratio", type=float, default=None,
        help=("Use a deterministic random token partition with this masked fraction; "
              "omit to use the training block-mask generator"),
    )
    parser.add_argument(
        "--future-visible-frames", type=int, default=None,
        help=("Use a causal future mask: expose every spatial token in the first N sampled "
              "frames and predict every token in the remaining frames. N must be divisible "
              "by the model tubelet size and cannot be combined with --mask-ratio."),
    )
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every discovered clip")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--min-load-percent", type=float, default=90.0,
                        help="Minimum matching key and parameter percentage for every JEPA module")
    parser.add_argument("--lpips", action="store_true", help="Also compute AlexNet LPIPS")
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Recompute cached per-model predicted latents")
    parser.add_argument("--hf-token", default=None,
                        help="Prefer setting HF_TOKEN in the environment")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="factorjepa-stage1-comparison")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="stage1-factorjepa-vs-vjepa-30clips")
    parser.add_argument("--wandb-max-videos", type=int, default=10)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parent.parent / candidate
    return candidate


def discover_clips(root: Path, limit: int) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    clips = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )
    if not clips:
        raise RuntimeError(f"no MP4 files found under {root}")
    if limit > 0:
        clips = clips[:limit]
    return clips


def clip_key(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_fixed_ratio_masks(keys: list[str], base_seed: int, n_total: int,
                            mask_ratio: float):
    """Create exact, deterministic per-clip random token partitions."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("--mask-ratio must be strictly between 0 and 1")
    n_pred = max(1, min(n_total - 1, int(round(n_total * mask_ratio))))
    masks = []
    seeds = []
    for key in keys:
        seed = validation_seed_for_clip(base_seed, key)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(n_total, generator=generator)
        m_pred = permutation[:n_pred].sort().values.unsqueeze(0)
        m_enc = permutation[n_pred:].sort().values.unsqueeze(0)
        masks.append((m_enc, m_pred))
        seeds.append(seed)
    return masks, seeds, {
        "mode": "deterministic_random_ratio",
        "requested_mask_ratio": float(mask_ratio),
        "n_total": int(n_total),
        "n_predicted": int(n_pred),
        "n_visible": int(n_total - n_pred),
        "actual_mask_ratio": float(n_pred / n_total),
        "actual_visible_ratio": float((n_total - n_pred) / n_total),
    }


def build_future_masks(keys: list[str], base_seed: int, token_grid: tuple[int, int, int],
                       tubelet_size: int, visible_frames: int):
    """Expose an initial temporal prefix and predict the complete spatial future."""
    t_tokens, h_tokens, w_tokens = (int(value) for value in token_grid)
    tubelet_size = int(tubelet_size)
    visible_frames = int(visible_frames)
    total_frames = t_tokens * tubelet_size
    if visible_frames <= 0 or visible_frames >= total_frames:
        raise ValueError(
            f"--future-visible-frames must be in [1, {total_frames - 1}], "
            f"got {visible_frames}"
        )
    if visible_frames % tubelet_size:
        raise ValueError(
            f"--future-visible-frames={visible_frames} must be divisible by "
            f"tubelet_size={tubelet_size}; a JEPA tubelet cannot be partially visible"
        )

    visible_t_tokens = visible_frames // tubelet_size
    tokens_per_time = h_tokens * w_tokens
    n_visible = visible_t_tokens * tokens_per_time
    n_total = t_tokens * tokens_per_time
    m_enc = torch.arange(0, n_visible, dtype=torch.long).unsqueeze(0)
    m_pred = torch.arange(n_visible, n_total, dtype=torch.long).unsqueeze(0)
    masks = [(m_enc.clone(), m_pred.clone()) for _ in keys]
    seeds = [validation_seed_for_clip(base_seed, key) for key in keys]
    return masks, seeds, {
        "mode": "causal_future",
        "requested_visible_frames": visible_frames,
        "effective_visible_frames": visible_t_tokens * tubelet_size,
        "first_predicted_frame": visible_t_tokens * tubelet_size,
        "total_frames": total_frames,
        "tubelet_size": tubelet_size,
        "tokens_per_temporal_plane": tokens_per_time,
        "visible_temporal_tokens": visible_t_tokens,
        "predicted_temporal_tokens": t_tokens - visible_t_tokens,
        "n_total": n_total,
        "n_predicted": n_total - n_visible,
        "n_visible": n_visible,
        "actual_mask_ratio": float((n_total - n_visible) / n_total),
        "actual_visible_ratio": float(n_visible / n_total),
    }


def safe_stem(key: str) -> str:
    stem = str(Path(key).with_suffix(""))
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "__", stem).strip("._-")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned or 'clip'}__{digest}"


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def cache_fingerprint(context: dict, key: str, mask_seed: int) -> str:
    payload = {**context, "clip_key": key, "mask_seed": int(mask_seed)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_raw_batch(path: Path, key: str, data_cfg: dict, model_cfg: dict) -> torch.Tensor:
    with tempfile.TemporaryDirectory(prefix="stage1_compare_decode_") as tmp_dir:
        decoded = decode_video_bytes(path.read_bytes(), tmp_dir, key, int(data_cfg["num_frames"]))
    if decoded is None:
        raise RuntimeError(f"failed to decode {path}")
    raw = resize_center_crop(decoded, int(model_cfg["crop_size"]))
    return raw.unsqueeze(0)


def validate_external_config(cfg: dict, checkpoint: dict):
    checkpoint_model = checkpoint["model_cfg"]
    mismatches = {
        key: {"config": cfg["model"].get(key), "checkpoint": checkpoint_model.get(key)}
        for key in CONTRACT_KEYS
        if cfg["model"].get(key) != checkpoint_model.get(key)
    }
    if int(cfg["data"]["num_frames"]) != int(checkpoint["data_cfg"]["num_frames"]):
        mismatches["num_frames"] = {
            "config": cfg["data"]["num_frames"],
            "checkpoint": checkpoint["data_cfg"]["num_frames"],
        }
    if mismatches:
        raise RuntimeError(
            "model/train configs do not match the trained decoder checkpoint:\n"
            + json.dumps(mismatches, indent=2)
        )


def load_decoder(checkpoint_path: Path, device: torch.device):
    payload = torch_load_cpu(checkpoint_path, weights_only=False)
    required = {
        "decoder_state_dict", "model_cfg", "data_cfg", "stage_cfg", "token_grid",
        "latent_shape", "cosmos_model_id", "cosmos_revision",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"Stage-1 checkpoint is missing fields: {missing}")
    model_cfg = payload["model_cfg"]
    stage_cfg = payload["stage_cfg"]
    decoder = Stage1JepaGridDecoder(
        embed_dim=int(model_cfg["embed_dim"]),
        n_levels=int(model_cfg["n_output_distillation"]),
        token_grid=tuple(payload["token_grid"]),
        latent_shape=tuple(payload["latent_shape"]),
        decoder_dim=int(stage_cfg["decoder_dim"]),
        decoder_depth=int(stage_cfg["decoder_depth"]),
        kernel_size=int(stage_cfg["decoder_kernel_size"]),
        dropout=float(stage_cfg["dropout"]),
    )
    decoder_state = payload.pop("decoder_state_dict")
    decoder.load_state_dict(decoder_state, strict=True)
    del decoder_state
    gc.collect()
    decoder.to(device).eval().requires_grad_(False)
    return decoder, payload


def strip_prefixes(state_dict: dict) -> dict:
    cleaned = {}
    for original, value in state_dict.items():
        key = original
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def extract_model_states(payload: dict, kind: str) -> tuple[dict, dict]:
    if kind == "factorjepa":
        encoder_keys = ("student", "student_state_dict")
    elif kind == "vjepa":
        encoder_keys = ("target_encoder", "encoder")
    else:
        raise ValueError(kind)
    encoder = next((payload[key] for key in encoder_keys if key in payload), None)
    predictor = payload.get("predictor")
    if encoder is None or predictor is None:
        raise KeyError(
            f"{kind} checkpoint requires encoder key in {encoder_keys} and 'predictor'; "
            f"found top-level keys={list(payload)[:20]}"
        )
    return strip_prefixes(encoder), strip_prefixes(predictor)


def infer_mask_token_count(predictor_state: dict, fallback: int) -> int:
    indices = []
    for key in predictor_state:
        match = re.search(r"(?:^|\.)mask_tokens\.(\d+)$", key)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return max(indices) + 1
    tensor = predictor_state.get("mask_tokens")
    if torch.is_tensor(tensor) and tensor.ndim >= 1:
        return int(tensor.shape[0])
    return int(fallback)


def load_state_with_report(module: torch.nn.Module, state_dict: dict, minimum_percent: float,
                           label: str) -> dict:
    module_state = module.state_dict()
    shape_mismatches = []
    matched_keys = []
    for key, value in state_dict.items():
        if key not in module_state:
            continue
        if tuple(value.shape) != tuple(module_state[key].shape):
            shape_mismatches.append({
                "key": key,
                "checkpoint": list(value.shape),
                "model": list(module_state[key].shape),
            })
        else:
            matched_keys.append(key)
    if shape_mismatches:
        raise RuntimeError(
            f"{label} has incompatible tensor shapes. This usually means ViT-G was used "
            f"instead of ViT-g, or the predictor config is wrong:\n"
            + json.dumps(shape_mismatches[:12], indent=2)
        )
    message = module.load_state_dict(state_dict, strict=False)
    key_percent = 100.0 * len(matched_keys) / max(len(module_state), 1)
    loaded_numel = sum(module_state[key].numel() for key in matched_keys)
    total_numel = sum(value.numel() for value in module_state.values())
    parameter_percent = 100.0 * loaded_numel / max(total_numel, 1)
    report = {
        "matched_keys": len(matched_keys),
        "total_model_keys": len(module_state),
        "key_percent": key_percent,
        "parameter_percent": parameter_percent,
        "missing_keys": list(message.missing_keys),
        "unexpected_keys": list(message.unexpected_keys),
    }
    print(
        f"Loaded {label}: {len(matched_keys)}/{len(module_state)} keys "
        f"({key_percent:.2f}%), {parameter_percent:.2f}% of parameters; "
        f"missing={len(message.missing_keys)} unexpected={len(message.unexpected_keys)}"
    )
    if key_percent < float(minimum_percent) or parameter_percent < float(minimum_percent):
        raise RuntimeError(
            f"{label} load is below {minimum_percent:.2f}%: "
            f"keys={key_percent:.2f}% parameters={parameter_percent:.2f}%"
        )
    return report


def load_jepa_pair(checkpoint_path: Path, kind: str, model_cfg: dict, data_cfg: dict,
                   device: torch.device, dtype: torch.dtype, minimum_load_percent: float):
    payload = torch_load_cpu(checkpoint_path, weights_only=False)
    encoder_state, predictor_state = extract_model_states(payload, kind)
    build_cfg = dict(model_cfg)
    build_cfg["use_activation_checkpointing"] = False
    inferred_mask_tokens = infer_mask_token_count(
        predictor_state, int(build_cfg["num_mask_tokens"]))
    if inferred_mask_tokens != int(build_cfg["num_mask_tokens"]):
        print(
            f"  {kind}: predictor checkpoint contains {inferred_mask_tokens} mask tokens; "
            f"overriding config value {build_cfg['num_mask_tokens']} for construction"
        )
        build_cfg["num_mask_tokens"] = inferred_mask_tokens
    student, predictor = build_student_predictor(build_cfg, data_cfg)
    student_minimum = max(float(model_cfg["min_student_load_pct"]), minimum_load_percent)
    predictor_minimum = max(float(model_cfg["min_predictor_load_pct"]), minimum_load_percent)
    report = {
        "checkpoint": str(checkpoint_path),
        "encoder_key_schema": "student" if kind == "factorjepa" else "target_encoder/encoder",
        "num_mask_tokens": inferred_mask_tokens,
        "student": load_state_with_report(
            student, encoder_state, student_minimum,
            f"{kind} encoder"),
        "predictor": load_state_with_report(
            predictor, predictor_state, predictor_minimum,
            f"{kind} predictor"),
    }
    student.to(device=device, dtype=dtype).eval().requires_grad_(False)
    predictor.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if hasattr(student, "return_hierarchical"):
        student.return_hierarchical = int(model_cfg["n_output_distillation"]) > 1
    del payload, encoder_state, predictor_state
    gc.collect()
    return student, predictor, report


def resolve_factorjepa_checkpoint(args) -> Path:
    if args.factorjepa_ckpt:
        path = resolve_path(args.factorjepa_ckpt)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import hf_hub_download

    print(
        f"Downloading FactorJEPA checkpoint: "
        f"{args.factorjepa_hf_repo}/{args.factorjepa_hf_filename}"
    )
    return Path(hf_hub_download(
        repo_id=args.factorjepa_hf_repo,
        repo_type=args.factorjepa_hf_repo_type,
        filename=args.factorjepa_hf_filename,
        token=os.environ.get("HF_TOKEN") or None,
    ))


def resolve_vjepa_checkpoint(args) -> Path:
    path = resolve_path(args.vjepa_ckpt)
    if path.is_file():
        return path
    if not args.vjepa_url:
        raise FileNotFoundError(path)
    print(f"Downloading vanilla V-JEPA checkpoint: {args.vjepa_url} -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(args.vjepa_url, str(path), progress=True)
    return path


def latent_cache_path(output_dir: Path, model_name: str, key: str) -> Path:
    return output_dir / "latent_cache" / model_name / f"{safe_stem(key)}.pt"


@torch.no_grad()
def run_backbone_pass(*, model_name: str, checkpoint_path: Path, clips: list[Path],
                      clips_root: Path, masks: dict, cfg: dict, decoder, device, dtype,
                      output_dir: Path, overwrite_cache: bool, minimum_load_percent: float,
                      mask_seeds: dict, cache_context: dict):
    student, predictor, load_report = load_jepa_pair(
        checkpoint_path, model_name, cfg["model"], cfg["data"], device, dtype,
        minimum_load_percent)
    latents = {}
    grid_records = {}
    for path in tqdm(clips, desc=f"{model_name} JEPA -> Stage-1 latent"):
        key = clip_key(path, clips_root)
        cache_path = latent_cache_path(output_dir, model_name, key)
        fingerprint = cache_fingerprint(cache_context, key, mask_seeds[key])
        if cache_path.is_file() and not overwrite_cache:
            cached = torch_load_cpu(cache_path, weights_only=True)
            if cached.get("fingerprint") == fingerprint:
                latents[key] = cached["pred_latent"].float()
                grid_records[key] = cached["grid_meta"]
                continue
            print(f"  Ignoring stale latent cache for {model_name}/{key}")
        raw_batch = load_raw_batch(path, key, cfg["data"], cfg["model"]).to(device)
        full_grid, token_types, grid_meta = build_full_jepa_grid(
            raw_batch, cfg, [], student, predictor, device, fixed_masks=masks[key])
        expected_width = int(cfg["model"]["embed_dim"]) * int(
            cfg["model"]["n_output_distillation"])
        expected_grid = tuple(decoder.token_grid)
        if tuple(grid_meta["token_grid"]) != expected_grid:
            raise RuntimeError(
                f"{key}: JEPA grid {grid_meta['token_grid']} != decoder grid {expected_grid}")
        if int(grid_meta["jepa_dim"]) != expected_width:
            raise RuntimeError(
                f"{key}: combined token width {grid_meta['jepa_dim']} != {expected_width}")
        pred_latent = decoder(full_grid, token_types).float().cpu()
        expected_latent = (1, *tuple(decoder.latent_shape))
        if tuple(pred_latent.shape) != expected_latent:
            raise RuntimeError(
                f"{key}: Stage-1 latent shape {tuple(pred_latent.shape)} != {expected_latent}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "pred_latent": pred_latent,
            "grid_meta": grid_meta,
            "fingerprint": fingerprint,
        }, cache_path)
        latents[key] = pred_latent
        grid_records[key] = grid_meta
        del raw_batch, full_grid, token_types, pred_latent
    del student, predictor
    cuda_cleanup()
    return latents, grid_records, load_report


def video_to_frames(video_unit_range: torch.Tensor) -> list[Image.Image]:
    video = (video_unit_range[0].detach().float().clamp(0, 1) * 255.0).byte()
    arrays = video.permute(1, 2, 3, 0).cpu().numpy()
    return [Image.fromarray(frame) for frame in arrays]


@torch.no_grad()
def decode_latents_native(pipe, latents: torch.Tensor) -> torch.Tensor:
    """Decode the VAE's native temporal length without repeating or fabricating frames."""
    latents = denormalize_from_cosmos_vae(pipe, latents)
    with torch.amp.autocast("cuda", enabled=False):
        decoded = pipe.vae.decode(latents.to(pipe.dtype), return_dict=False)[0]
    if not torch.isfinite(decoded).all():
        raise RuntimeError("Cosmos VAE decode produced NaN/Inf values")
    return decoded.float().clamp(-1, 1).add(1.0).mul(0.5)


def pixel_token_types(raw_batch: torch.Tensor, mask_pair: tuple[torch.Tensor, torch.Tensor],
                      model_cfg: dict) -> torch.Tensor:
    """Expand JEPA token indices into the video pixel grid without approximating the mask."""
    _, _, num_frames, height, width = raw_batch.shape
    tubelet = int(model_cfg["tubelet_size"])
    patch = int(model_cfg["patch_size"])
    if num_frames % tubelet or height % patch or width % patch:
        raise RuntimeError(
            f"video shape {(num_frames, height, width)} is not divisible by "
            f"tubelet/patch {(tubelet, patch, patch)}"
        )
    token_grid = (num_frames // tubelet, height // patch, width // patch)
    n_total = token_grid[0] * token_grid[1] * token_grid[2]
    visible = mask_pair[0].reshape(-1).long().cpu()
    predicted = mask_pair[1].reshape(-1).long().cpu()
    for label, indices in (("visible", visible), ("predicted", predicted)):
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= n_total):
            raise RuntimeError(f"{label} mask index is outside token grid 0..{n_total - 1}")
        if torch.unique(indices).numel() != indices.numel():
            raise RuntimeError(f"{label} mask contains duplicate token indices")
    if torch.isin(visible, predicted).any():
        raise RuntimeError("visible and predicted JEPA masks overlap")

    # 0=visible to encoder, 1=masked/predicted target, 2=unused by this mask pair.
    token_types = torch.full((n_total,), 2, dtype=torch.uint8)
    token_types[visible] = 0
    token_types[predicted] = 1
    return (
        token_types.reshape(token_grid)
        .repeat_interleave(tubelet, dim=0)
        .repeat_interleave(patch, dim=1)
        .repeat_interleave(patch, dim=2)
    )


def mask_visualization_frames(
        raw_batch: torch.Tensor, mask_pair: tuple[torch.Tensor, torch.Tensor], model_cfg: dict,
        output_frames: int = None,
        ) -> list[Image.Image]:
    """Render encoder-visible pixels and the exact visible/predicted token overlay."""
    video = raw_batch[0].detach().float().clamp(0, 1).cpu()
    token_types = pixel_token_types(raw_batch, mask_pair, model_cfg)
    if output_frames is not None:
        if output_frames <= 0 or output_frames > video.shape[1]:
            raise ValueError(
                f"output_frames must be in [1, {video.shape[1]}], got {output_frames}")
        video = video[:, :output_frames]
        token_types = token_types[:output_frames]
    visible = token_types.eq(0).unsqueeze(0)
    predicted = token_types.eq(1).unsqueeze(0)

    # The encoder receives token indices, not a black-pixel video. This panel is a spatial
    # visualization of which tubelets survive the encoder mask.
    visible_only = torch.where(visible, video, video * 0.06)
    colors = torch.zeros_like(video)
    colors[:] = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    colors = torch.where(visible, torch.tensor([0.10, 0.85, 0.25]).view(3, 1, 1, 1), colors)
    colors = torch.where(predicted, torch.tensor([0.95, 0.12, 0.12]).view(3, 1, 1, 1), colors)
    overlay = video * 0.62 + colors * 0.38

    return comparison_frames([
        ("Encoder-visible tokens", video_to_frames(visible_only.unsqueeze(0))),
        ("Green visible | Red predicted", video_to_frames(overlay.unsqueeze(0))),
    ])


def labeled_frame(frame: Image.Image, label: str, label_height: int = 28) -> Image.Image:
    canvas = Image.new("RGB", (frame.width, frame.height + label_height), color=(20, 20, 20))
    canvas.paste(frame.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(canvas)
    box = draw.textbbox((0, 0), label)
    text_width = box[2] - box[0]
    draw.text(((frame.width - text_width) // 2, 7), label, fill=(245, 245, 245))
    return canvas


def comparison_frames(groups: list[tuple[str, list[Image.Image]]]) -> list[Image.Image]:
    lengths = {len(frames) for _, frames in groups}
    if len(lengths) != 1:
        raise RuntimeError(f"comparison videos have unequal frame counts: {lengths}")
    panels = []
    for index in range(next(iter(lengths))):
        labeled = [labeled_frame(frames[index], label) for label, frames in groups]
        panel = Image.new("RGB", (sum(frame.width for frame in labeled), labeled[0].height))
        left = 0
        for frame in labeled:
            panel.paste(frame, (left, 0))
            left += frame.width
        panels.append(panel)
    return panels


def compute_metrics(pred_unit: torch.Tensor, target_unit: torch.Tensor,
                    pred_latent: torch.Tensor, target_latent: torch.Tensor,
                    lpips_model) -> dict:
    pixel_delta = pred_unit.float() - target_unit.float()
    pixel_mse = pixel_delta.pow(2).mean()
    pred_dt = pred_unit[:, :, 1:] - pred_unit[:, :, :-1]
    target_dt = target_unit[:, :, 1:] - target_unit[:, :, :-1]
    values = {
        "latent_mse": F.mse_loss(pred_latent.float(), target_latent.float()).item(),
        "latent_mae": F.l1_loss(pred_latent.float(), target_latent.float()).item(),
        "pixel_mse": pixel_mse.item(),
        "pixel_mae": pixel_delta.abs().mean().item(),
        "psnr_db": float(-10.0 * torch.log10(pixel_mse.clamp_min(1.0e-12)).item()),
        "temporal_difference_mae": F.l1_loss(pred_dt, target_dt).item(),
    }
    if lpips_model is not None:
        pred_lpips = flatten_lpips_frames(pred_unit * 2.0 - 1.0, 4)
        target_lpips = flatten_lpips_frames(target_unit * 2.0 - 1.0, 4)
        values["lpips"] = lpips_model(pred_lpips.float(), target_lpips.float()).mean().item()
    return values


def aggregate_metrics(rows: list[dict]) -> dict:
    result = {}
    for model_name in ("factorjepa", "vjepa"):
        model_rows = [row for row in rows if row["model"] == model_name]
        numeric_keys = sorted(
            key for key, value in model_rows[0].items()
            if key not in {"model", "clip_key"} and isinstance(value, (int, float)))
        result[model_name] = {
            key: float(sum(float(row[key]) for row in model_rows) / len(model_rows))
            for key in numeric_keys
        }
    return result


def write_metrics_csv(path: Path, rows: list[dict]):
    columns = ["clip_key", "model"] + sorted(set().union(*(row.keys() for row in rows)) - {
        "clip_key", "model"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def init_wandb(args, public_config: dict):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("--wandb requires: python -m pip install wandb") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=public_config,
    )


def log_wandb(run, rows: list[dict], summary: dict, panel_paths: list[Path],
              mask_paths: list[Path], max_videos: int, output_dir: Path):
    if run is None:
        return
    import wandb

    columns = ["clip_key", "model"] + sorted(set().union(*(row.keys() for row in rows)) - {
        "clip_key", "model"})
    payload = {
        "comparison/metrics_per_clip": wandb.Table(
            columns=columns, data=[[row.get(column) for column in columns] for row in rows])
    }
    for model_name, metrics in summary.items():
        payload.update({f"comparison/{model_name}/{key}": value for key, value in metrics.items()})
    for index, path in enumerate(panel_paths[:max(0, max_videos)]):
        payload[f"comparison/videos/{index:02d}_{path.stem}"] = wandb.Video(str(path), format="mp4")
    for index, path in enumerate(mask_paths[:max(0, max_videos)]):
        payload[f"comparison/masks/{index:02d}_{path.stem}"] = wandb.Video(str(path), format="mp4")
    run.log(payload)
    artifact = wandb.Artifact(f"{run.id}-stage1-comparison", type="evaluation")
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact, aliases=["latest"])


@torch.no_grad()
def main():
    args = parse_args()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        stage1_train.HF_TOKEN = args.hf_token
    check_gpu()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if not 0.0 < args.min_load_percent <= 100.0:
        raise ValueError("--min-load-percent must be in (0, 100]")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_root = resolve_path(args.clips_dir)
    clips = discover_clips(clips_root, args.limit)

    decoder_path = resolve_path(args.decoder_ckpt)
    if not decoder_path.is_file():
        raise FileNotFoundError(decoder_path)
    cfg = load_merged_config(args.model_config, args.train_config)
    decoder, decoder_payload = load_decoder(decoder_path, device)
    validate_external_config(cfg, decoder_payload)
    cfg["model"] = dict(decoder_payload["model_cfg"])
    cfg["model"]["use_activation_checkpointing"] = False
    cfg["data"] = dict(decoder_payload["data_cfg"])
    cfg["mixed_precision"] = {"enabled": True, "dtype": args.dtype}

    expected_width = int(cfg["model"]["embed_dim"]) * int(
        cfg["model"]["n_output_distillation"])
    if int(cfg["model"]["embed_dim"]) != 1408 or expected_width != 5632:
        raise RuntimeError(
            "This comparison expects the matched V-JEPA 2.1 ViT-g contract: "
            f"embed_dim=1408 and combined width=5632, got {cfg['model']['embed_dim']} "
            f"and {expected_width}. Do not use the 1664-dim ViT-G checkpoint."
        )

    keys = [clip_key(path, clips_root) for path in clips]
    if args.future_visible_frames is not None and args.mask_ratio is not None:
        raise ValueError("--future-visible-frames and --mask-ratio are mutually exclusive")
    if args.future_visible_frames is not None:
        mask_list, mask_seeds, mask_contract = build_future_masks(
            keys,
            int(args.mask_seed),
            tuple(decoder.token_grid),
            int(cfg["model"]["tubelet_size"]),
            int(args.future_visible_frames),
        )
    elif args.mask_ratio is None:
        mask_generators = build_mask_generators(cfg)
        mask_list, mask_seeds = build_per_clip_validation_masks(
            mask_generators, keys, int(args.mask_seed))
        sample_enc, sample_pred = mask_list[0]
        mask_contract = {
            "mode": "training_block_mask",
            "requested_mask_ratio": None,
            "n_total": int(torch.tensor(decoder.token_grid).prod().item()),
            "n_predicted": int(sample_pred.reshape(-1).numel()),
            "n_visible": int(sample_enc.reshape(-1).numel()),
        }
        mask_contract["actual_mask_ratio"] = (
            mask_contract["n_predicted"] / mask_contract["n_total"])
        mask_contract["actual_visible_ratio"] = (
            mask_contract["n_visible"] / mask_contract["n_total"])
        del mask_generators
    else:
        n_total = int(torch.tensor(decoder.token_grid).prod().item())
        mask_list, mask_seeds, mask_contract = build_fixed_ratio_masks(
            keys, int(args.mask_seed), n_total, float(args.mask_ratio))
    masks = dict(zip(keys, mask_list))
    seeds = dict(zip(keys, mask_seeds))

    factor_path = resolve_factorjepa_checkpoint(args)
    vjepa_path = resolve_vjepa_checkpoint(args)
    print(
        f"Comparing {len(clips)} clips with deterministic per-clip masks: "
        f"mode={mask_contract['mode']} predicted={mask_contract['n_predicted']}/"
        f"{mask_contract['n_total']} ({100.0 * mask_contract['actual_mask_ratio']:.2f}%) "
        f"visible={mask_contract['n_visible']}/"
        f"{mask_contract['n_total']} ({100.0 * mask_contract['actual_visible_ratio']:.2f}%)"
    )
    print(f"Stage-1 decoder: {decoder_path} (step={decoder_payload.get('step', 'unknown')})")
    print(f"FactorJEPA: {factor_path}")
    print(f"V-JEPA 2.1 ViT-g: {vjepa_path}")

    all_latents = {}
    grid_records = {}
    load_reports = {}
    for model_name, checkpoint_path in (("factorjepa", factor_path), ("vjepa", vjepa_path)):
        cache_context = {
            "schema_version": 2,
            "model": model_name,
            "model_checkpoint": file_identity(checkpoint_path),
            "decoder_checkpoint": file_identity(decoder_path),
            "decoder_step": decoder_payload.get("step"),
            "dtype": args.dtype,
            "model_contract": {key: cfg["model"].get(key) for key in CONTRACT_KEYS},
            "num_frames": int(cfg["data"]["num_frames"]),
            "mask_contract": mask_contract,
        }
        latents, records, report = run_backbone_pass(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            clips=clips,
            clips_root=clips_root,
            masks=masks,
            cfg=cfg,
            decoder=decoder,
            device=device,
            dtype=dtype,
            output_dir=output_dir,
            overwrite_cache=args.overwrite_cache,
            minimum_load_percent=args.min_load_percent,
            mask_seeds=seeds,
            cache_context=cache_context,
        )
        all_latents[model_name] = latents
        grid_records[model_name] = records
        load_reports[model_name] = report

    stage_cfg = decoder_payload["stage_cfg"]
    cosmos = load_cosmos_vae(
        decoder_payload["cosmos_model_id"],
        decoder_payload["cosmos_revision"],
        dtype,
        device,
        vae_subfolder=stage_cfg.get("cosmos_vae_subfolder", "vae"),
        enable_tiling=stage_cfg.get("cosmos_vae_enable_tiling", False),
        enable_slicing=stage_cfg.get("cosmos_vae_enable_slicing", False),
    )
    latent_t = int(decoder.latent_shape[1])
    native_output_frames = (latent_t - 1) * int(cosmos.temporal_scale) + 1
    input_frames = int(cfg["data"]["num_frames"])
    if native_output_frames > input_frames:
        raise RuntimeError(
            f"native Cosmos output has {native_output_frames} frames but JEPA input has "
            f"only {input_frames}"
        )
    causal_mask = mask_contract.get("mode") == "causal_future"
    if causal_mask:
        displayed_visible_frames = min(
            int(mask_contract["effective_visible_frames"]), native_output_frames)
        displayed_predicted_frames = native_output_frames - displayed_visible_frames
        internal_predicted_frames = (
            int(mask_contract["total_frames"])
            - int(mask_contract["effective_visible_frames"])
        )
        undisplayed_predicted_frames = max(
            0, internal_predicted_frames - displayed_predicted_frames)
    else:
        displayed_visible_frames = None
        displayed_predicted_frames = None
        undisplayed_predicted_frames = None
    visualization_contract = {
        "jepa_input_frames": input_frames,
        "cosmos_latent_temporal_steps": latent_t,
        "cosmos_native_output_frames": native_output_frames,
        "displayed_visible_frames": displayed_visible_frames,
        "displayed_predicted_frames": displayed_predicted_frames,
        "undisplayed_internal_predicted_frames": undisplayed_predicted_frames,
        "temporal_alignment": "native_cosmos_decode_then_reference_prefix_crop",
    }
    alignment_message = (
        "Native Cosmos temporal alignment: "
        f"JEPA input={input_frames} frames, latent_t={latent_t}, "
        f"export={native_output_frames} frames"
    )
    if causal_mask:
        alignment_message += (
            f" ({displayed_visible_frames} visible + {displayed_predicted_frames} predicted); "
            f"{undisplayed_predicted_frames} internal predicted frame(s) are not renderable"
        )
    print(alignment_message)
    lpips_model = None
    if args.lpips:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("--lpips requires: python -m pip install lpips") from exc
        lpips_model = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)

    rows = []
    panel_paths = []
    mask_paths = []
    clip_records = []
    for path in tqdm(clips, desc="Cosmos VAE decode + metrics + export"):
        key = clip_key(path, clips_root)
        stem = safe_stem(key)
        raw_batch = load_raw_batch(path, key, cfg["data"], cfg["model"]).to(device)
        target_latent = encode_video_latents(cosmos, raw_batch, dtype).float()
        target_video = raw_batch[:, :, :native_output_frames].float().clamp(0, 1)
        roundtrip = decode_latents_native(cosmos, target_latent)
        if roundtrip.shape[2] != native_output_frames:
            raise RuntimeError(
                f"Cosmos target decode returned {roundtrip.shape[2]} frames; "
                f"expected native length {native_output_frames}"
            )
        frame_groups = [
            (f"Sampled input (first {native_output_frames})", video_to_frames(target_video)),
            ("Cosmos VAE roundtrip", video_to_frames(roundtrip)),
        ]
        model_metrics = {}
        for model_name, label in (("factorjepa", "FactorJEPA"), ("vjepa", "V-JEPA 2.1 ViT-g")):
            pred_latent = all_latents[model_name][key].to(device)
            pred_video = decode_latents_native(cosmos, pred_latent)
            if pred_video.shape[2] != native_output_frames:
                raise RuntimeError(
                    f"{model_name} Cosmos decode returned {pred_video.shape[2]} frames; "
                    f"expected native length {native_output_frames}"
                )
            metrics = compute_metrics(
                pred_video, target_video, pred_latent, target_latent, lpips_model)
            model_metrics[model_name] = metrics
            rows.append({"clip_key": key, "model": model_name, **metrics})
            frames = video_to_frames(pred_video)
            model_path = output_dir / "videos" / model_name / f"{stem}.mp4"
            export_video(frames, model_path, args.fps)
            frame_groups.append((label, frames))
            del pred_latent, pred_video

        reference_path = output_dir / "videos" / "reference" / f"{stem}.mp4"
        roundtrip_path = output_dir / "videos" / "cosmos_vae_roundtrip" / f"{stem}.mp4"
        panel_path = output_dir / "videos" / "comparison" / f"{stem}.mp4"
        mask_path = output_dir / "videos" / "masks" / f"{stem}.mp4"
        export_video(frame_groups[0][1], reference_path, args.fps)
        export_video(frame_groups[1][1], roundtrip_path, args.fps)
        export_video(comparison_frames(frame_groups), panel_path, args.fps)
        export_video(mask_visualization_frames(
                         raw_batch, masks[key], cfg["model"], native_output_frames),
                     mask_path, args.fps)
        panel_paths.append(panel_path)
        mask_paths.append(mask_path)
        clip_records.append({
            "clip_key": key,
            "source_path": str(path),
            "mask_seed": seeds[key],
            "exported_frames": native_output_frames,
            "grid": grid_records["factorjepa"][key],
            "factorjepa_metrics": model_metrics["factorjepa"],
            "vjepa_metrics": model_metrics["vjepa"],
            "comparison_video": str(panel_path),
            "mask_video": str(mask_path),
        })
        del raw_batch, target_latent, target_video, roundtrip

    metrics_path = output_dir / "metrics_per_clip.csv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "inference_manifest.json"
    reports_path = output_dir / "checkpoint_load_reports.json"
    write_metrics_csv(metrics_path, rows)
    summary = aggregate_metrics(rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    reports_path.write_text(json.dumps(load_reports, indent=2) + "\n")
    manifest = {
        "comparison_type": "shared FactorJEPA-trained Stage-1 decoder transfer",
        "scientific_caveat": (
            "A worse vanilla V-JEPA result can reflect decoder feature-space mismatch; "
            "a fully controlled comparison trains one identical decoder per frozen backbone."
        ),
        "num_clips": len(clips),
        "decoder_checkpoint": str(decoder_path),
        "decoder_step": decoder_payload.get("step"),
        "factorjepa_checkpoint": str(factor_path),
        "vjepa_checkpoint": str(vjepa_path),
        "model_contract": {
            "architecture": cfg["model"]["arch"],
            "input_video": [1, 3, int(cfg["data"]["num_frames"]),
                            int(cfg["model"]["crop_size"]), int(cfg["model"]["crop_size"])],
            "token_grid": list(decoder.token_grid),
            "combined_token_width": expected_width,
            "cosmos_latent": [1, *list(decoder.latent_shape)],
        },
        "mask_base_seed": int(args.mask_seed),
        "mask_contract": mask_contract,
        "visualization_contract": visualization_contract,
        "fps": int(args.fps),
        "lpips_enabled": bool(args.lpips),
        "clips": clip_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    public_config = {
        "decoder_checkpoint": str(decoder_path),
        "decoder_step": decoder_payload.get("step"),
        "factorjepa_checkpoint": str(factor_path),
        "vjepa_checkpoint": str(vjepa_path),
        "num_clips": len(clips),
        "mask_seed": args.mask_seed,
        "mask_contract": mask_contract,
        "visualization_contract": visualization_contract,
        "lpips": args.lpips,
    }
    run = init_wandb(args, public_config)
    try:
        log_wandb(
            run, rows, summary, panel_paths, mask_paths, args.wandb_max_videos, output_dir)
    finally:
        if run is not None:
            run.finish()

    print(json.dumps(summary, indent=2))
    print(f"Saved comparison videos: {output_dir / 'videos' / 'comparison'}")
    print(f"Saved mask videos: {output_dir / 'videos' / 'masks'}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL (stage1-factorjepa-vs-vjepa): {type(exc).__name__}: {exc}")
        raise
