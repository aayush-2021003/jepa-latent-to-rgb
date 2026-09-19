import argparse
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_ENDPOINT"] = os.environ.get("FACTORJEPA_HF_ENDPOINT", "https://huggingface.co")
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = os.environ["HF_ENDPOINT"]

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import stage1_jepa_decoder_train as stage1_train
from stage1_jepa_decoder_infer import load_decoder as load_stage1_decoder
from stage1_jepa_decoder_infer import resize_center_crop
from stage1_jepa_decoder_train import (
    DEFAULT_SOURCE_HF_FILENAME,
    DEFAULT_SOURCE_HF_REPO,
    DEFAULT_SOURCE_CKPT,
    DENSEWORLD_DATASET_ID,
    build_validation_summary,
    build_per_clip_validation_masks,
    build_indexed_split,
    build_full_jepa_grid,
    build_lpips_model,
    decode_latents_to_frames,
    decode_latents_for_loss,
    encode_video_latents,
    flatten_lpips_frames,
    init_wandb,
    load_cosmos_vae,
    load_frozen_jepa,
    load_or_create_random_split,
    load_validation_set,
    periodic_checkpoint_due,
    producer_thread,
    print_cuda_memory,
    raw_batch_to_frames,
    resolve_source_ckpt,
    resolve_validation_every_n_steps,
    sample_fixed_validation_masks,
    save_validation_references,
    wandb_log_artifact,
    wandb_log_artifact_many,
    wandb_log_metrics,
    wandb_log_validation_summary,
    wandb_log_video,
    wandb_log_videos,
    wandb_save_file,
    charbonnier_loss,
    spatial_gradient_loss,
    temporal_difference_loss,
    torch_load_cpu,
)
from utils.cache_policy import add_cache_policy_arg, resolve_cache_policy_interactive, wipe_output_dir
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
from utils.config import check_gpu, load_merged_config
from utils.config import get_pipeline_config
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.gpu_batch import cuda_cleanup
from utils.progress import make_pbar
from utils.training import build_mask_generators, load_config
from utils.video_io import decode_video_bytes


CHECKPOINT_LATEST = "stage2_latent_refiner_latest.pt"
CHECKPOINT_FINAL = "stage2_latent_refiner.pt"
CHECKPOINT_STEP_TEMPLATE = "stage2_latent_refiner_step_{step:07d}.pt"


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


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


def load_first_clip_from_local_data(local_data: str, cfg: dict, max_tar_files: int) -> tuple[str, torch.Tensor]:
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]
    clip_q, stop_event, reader = iter_clips_parallel(local_data, max_tar_files=max_tar_files)
    try:
        clip_key, mp4_bytes = None, None
        while True:
            item = clip_q.get(timeout=120)
            if item is None:
                break
            clip_key, mp4_bytes = item
            if mp4_bytes:
                break
        if not clip_key or not mp4_bytes:
            raise RuntimeError("no usable MP4 found in local data")
    finally:
        stop_event.set()
        reader.join(timeout=5)

    with tempfile.TemporaryDirectory(prefix="stage2_overfit_decode_") as tmp_dir:
        decoded = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames)
    if decoded is None:
        raise RuntimeError(f"failed to decode {clip_key}")
    raw_clip = resize_center_crop(decoded, crop_size)
    return clip_key, raw_clip.unsqueeze(0)


def load_clip_by_key(local_data: str, clip_key: str, cfg: dict, max_tar_files: int) -> tuple[str, torch.Tensor]:
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]
    clip_q, stop_event, reader = iter_clips_parallel(
        local_data, subset_keys={clip_key}, max_tar_files=max_tar_files)
    try:
        mp4_bytes = None
        while True:
            item = clip_q.get(timeout=120)
            if item is None:
                break
            found_key, found_bytes = item
            if found_key == clip_key:
                mp4_bytes = found_bytes
                break
        if not mp4_bytes:
            raise FileNotFoundError(f"clip key not found: {clip_key}")
    finally:
        stop_event.set()
        reader.join(timeout=5)

    with tempfile.TemporaryDirectory(prefix="stage2_overfit_decode_") as tmp_dir:
        decoded = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames)
    if decoded is None:
        raise RuntimeError(f"failed to decode {clip_key}")
    raw_clip = resize_center_crop(decoded, crop_size)
    return clip_key, raw_clip.unsqueeze(0)


def load_input_sample(args, cfg: dict, max_tar_files: int) -> tuple[str, torch.Tensor]:
    if args.input_mp4:
        path = Path(args.input_mp4)
        if not path.exists():
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory(prefix="stage2_overfit_decode_") as tmp_dir:
            decoded = decode_video_bytes(path.read_bytes(), tmp_dir, path.name, cfg["data"]["num_frames"])
        if decoded is None:
            raise RuntimeError(f"failed to decode {path}")
        raw_clip = resize_center_crop(decoded, cfg["data"]["crop_size"])
        return path.stem, raw_clip.unsqueeze(0)
    if args.clip_key:
        return load_clip_by_key(args.local_data, args.clip_key, cfg, max_tar_files)
    return load_first_clip_from_local_data(args.local_data, cfg, max_tar_files)


class JepaConditionPool(nn.Module):
    def __init__(self, embed_dim: int, n_levels: int, cond_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_levels = n_levels
        self.level_norm = nn.LayerNorm(embed_dim)
        self.level_logits = nn.Parameter(torch.zeros(n_levels))
        self.token_type = nn.Embedding(3, embed_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, full_grid: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
        if full_grid.shape[-1] == self.embed_dim:
            x = self.level_norm(full_grid)
        else:
            expected = self.embed_dim * self.n_levels
            if full_grid.shape[-1] != expected:
                raise RuntimeError(f"unexpected JEPA width {full_grid.shape[-1]}, expected {self.embed_dim} or {expected}")
            bsz, n_tokens, _ = full_grid.shape
            x = full_grid.view(bsz, n_tokens, self.n_levels, self.embed_dim)
            x = self.level_norm(x)
            weights = torch.softmax(self.level_logits, dim=0).view(1, 1, self.n_levels, 1)
            x = (x * weights).sum(dim=2)
        x = x + self.token_type(token_types).to(x.dtype)
        return self.out(x.mean(dim=1))


class RefinerBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        groups = group_count(channels)
        self.cond = nn.Linear(cond_dim, channels * 2)
        self.conv = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size, padding=pad),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size, padding=pad),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        scale = scale.view(scale.shape[0], scale.shape[1], 1, 1, 1)
        shift = shift.view(shift.shape[0], shift.shape[1], 1, 1, 1)
        return x + self.conv(x * (1 + scale) + shift)


class Stage2LatentRefiner(nn.Module):
    def __init__(self, *, latent_channels: int, embed_dim: int, n_levels: int,
                 hidden_dim: int, depth: int, kernel_size: int, dropout: float,
                 cond_dim: int, residual_scale: float):
        super().__init__()
        self.latent_channels = latent_channels
        self.residual_scale = residual_scale
        self.jepa_pool = JepaConditionPool(embed_dim, n_levels, cond_dim)
        self.in_proj = nn.Conv3d(latent_channels, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            RefinerBlock(hidden_dim, cond_dim, kernel_size, dropout)
            for _ in range(depth)
        ])
        self.out_norm = nn.GroupNorm(group_count(hidden_dim), hidden_dim)
        self.out = nn.Conv3d(hidden_dim, latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, coarse_latent: torch.Tensor,
                full_grid: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
        amp_enabled = full_grid.is_cuda and full_grid.dtype in (torch.bfloat16, torch.float16)
        amp_dtype = full_grid.dtype if amp_enabled else torch.bfloat16
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            cond = self.jepa_pool(full_grid, token_types)
            x = self.in_proj(coarse_latent)
            for block in self.blocks:
                x = block(x, cond)
            residual = self.out(F.silu(self.out_norm(x)))
            return coarse_latent + self.residual_scale * residual


def compute_loss(*, pred_latent: torch.Tensor, target_latent: torch.Tensor,
                 raw_batch: torch.Tensor, pipe, dtype: torch.dtype,
                 num_frames: int, loss_cfg: dict, lpips_model,
                 step: int) -> tuple[torch.Tensor, dict]:
    mse = F.mse_loss(pred_latent.float(), target_latent.detach().float())
    l1 = F.l1_loss(pred_latent.float(), target_latent.detach().float())
    total = loss_cfg["latent_mse_weight"] * mse + loss_cfg["latent_l1_weight"] * l1
    terms = {
        "loss_total": float(total.item()),
        "loss_latent_mse": float(mse.item()),
        "loss_latent_l1": float(l1.item()),
    }

    decoded_every = int(loss_cfg.get("decoded_loss_every_n_steps", 1))
    decoded_weight_sum = (
        loss_cfg["rgb_l1_weight"]
        + loss_cfg["spatial_gradient_weight"]
        + loss_cfg["temporal_difference_weight"]
        + loss_cfg["lpips_weight"]
    )
    run_decoded_losses = decoded_every > 0 and step % decoded_every == 0 and decoded_weight_sum > 0
    rgb_l1 = pred_latent.new_tensor(0.0)
    spatial = pred_latent.new_tensor(0.0)
    temporal = pred_latent.new_tensor(0.0)
    lpips_loss = pred_latent.new_tensor(0.0)

    if run_decoded_losses:
        pred_video = decode_latents_for_loss(pipe, pred_latent, dtype, num_frames)
        target_video = (raw_batch * 2.0 - 1.0).detach()
        if pred_video.shape != target_video.shape:
            raise RuntimeError(
                f"decoded Stage-2 video {tuple(pred_video.shape)} != "
                f"target video {tuple(target_video.shape)}"
            )
        pred_video = pred_video.float().clamp(-1, 1)
        target_video = target_video.float().clamp(-1, 1)
        eps = float(loss_cfg["charbonnier_eps"])
        rgb_l1 = charbonnier_loss(pred_video, target_video, eps)
        spatial = spatial_gradient_loss(pred_video, target_video, eps)
        temporal = temporal_difference_loss(pred_video, target_video, eps)
        total = (
            total
            + loss_cfg["rgb_l1_weight"] * rgb_l1
            + loss_cfg["spatial_gradient_weight"] * spatial
            + loss_cfg["temporal_difference_weight"] * temporal
        )
        if loss_cfg["lpips_weight"] > 0:
            if lpips_model is None:
                raise RuntimeError("Stage-2 LPIPS loss is enabled but lpips_model is None")
            pred_lpips = flatten_lpips_frames(pred_video, loss_cfg["lpips_max_frames"])
            target_lpips = flatten_lpips_frames(target_video, loss_cfg["lpips_max_frames"])
            lpips_loss = lpips_model(pred_lpips.float(), target_lpips.float()).mean()
            total = total + loss_cfg["lpips_weight"] * lpips_loss

    terms.update({
        "loss_total": float(total.item()),
        "loss_rgb_l1": float(rgb_l1.item()),
        "loss_spatial_gradient": float(spatial.item()),
        "loss_temporal_difference": float(temporal.item()),
        "loss_lpips": float(lpips_loss.item()),
        "decoded_losses_ran": bool(run_decoded_losses),
    })
    return total, terms


def save_checkpoint(path: Path, refiner: Stage2LatentRefiner, optimizer, scheduler,
                    step: int, stage2_cfg: dict, stage1_ckpt: str, source_ckpt: str,
                    model_cfg: dict, data_cfg: dict, latent_shape: tuple[int, ...],
                    include_optimizer: bool = True) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "refiner_state_dict": refiner.state_dict(),
        "step": step,
        "stage2_cfg": stage2_cfg,
        "stage1_ckpt": stage1_ckpt,
        "source_ckpt": source_ckpt,
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
        "latent_shape": list(latent_shape),
        "has_optimizer": include_optimizer,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"  WARN: failed to save checkpoint {path}: {exc}")
        return False
    return True


def step_checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / CHECKPOINT_STEP_TEMPLATE.format(step=step)


@torch.no_grad()
def run_validation(step: int, output_dir: Path, stage2_cfg: dict, cfg: dict, pipe,
                   refiner, raw_batch: torch.Tensor, full_grid: torch.Tensor,
                   token_types: torch.Tensor, coarse_latent: torch.Tensor,
                   target_latent: torch.Tensor, clip_key: str, dtype: torch.dtype,
                   validation_mask_seed: int, lpips_model):
    refiner.eval()
    val_dir = output_dir / "validation"
    fps = stage2_cfg["validation"]["fps"]

    refined_latent = refiner(coarse_latent, full_grid, token_types)
    _, refined_metrics = compute_loss(
        pred_latent=refined_latent,
        target_latent=target_latent,
        raw_batch=raw_batch,
        pipe=pipe,
        dtype=dtype,
        num_frames=cfg["data"]["num_frames"],
        loss_cfg=stage2_cfg["loss"],
        lpips_model=lpips_model,
        step=0,
    )
    coarse_mse = F.mse_loss(coarse_latent.float(), target_latent.float()).item()
    coarse_l1 = F.l1_loss(coarse_latent.float(), target_latent.float()).item()
    record = {
        "validation_index": 0,
        "clip_key": clip_key,
        "validation_mask_base_seed": stage2_cfg["validation"]["seed"],
        "validation_mask_seed": validation_mask_seed,
        **refined_metrics,
        "stage1_coarse_latent_mse": float(coarse_mse),
        "stage1_coarse_latent_l1": float(coarse_l1),
        "latent_mse_improvement": float(coarse_mse - refined_metrics["loss_latent_mse"]),
    }
    outputs = {
        "reference_input": raw_batch_to_frames(raw_batch.detach().cpu()),
        "stage1_coarse": decode_latents_to_frames(pipe, coarse_latent, dtype, cfg["data"]["num_frames"]),
        f"stage2_refined_step_{step:07d}": decode_latents_to_frames(pipe, refined_latent, dtype, cfg["data"]["num_frames"]),
        "cosmos_vae_roundtrip": decode_latents_to_frames(pipe, target_latent, dtype, cfg["data"]["num_frames"]),
    }
    paths = []
    for name, frames in outputs.items():
        path = val_dir / f"{name}.mp4"
        export_video(frames, path, fps)
        paths.append(path)
    meta_path = val_dir / f"stage2_refined_step_{step:07d}.json"
    meta_path.write_text(json.dumps({
        "step": step,
        "clip_key": clip_key,
        "validation_mask_base_seed": stage2_cfg["validation"]["seed"],
        "validation_mask_seed": validation_mask_seed,
        "coarse_latent_shape": list(coarse_latent.shape),
        "target_latent_shape": list(target_latent.shape),
        "refined_latent_shape": list(refined_latent.shape),
        "metrics": record,
    }, indent=2) + "\n")
    paths.append(meta_path)
    summary = build_validation_summary(step, [record])
    metrics_path = val_dir / f"validation_metrics_step_{step:07d}.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    paths.append(metrics_path)
    refiner.train()
    print(f"\n[validation] saved Stage-2 videos under {val_dir}")
    return paths, summary


@torch.no_grad()
def run_validation_set(step: int, output_dir: Path, stage2_cfg: dict, cfg: dict, pipe,
                       refiner, stage1_decoder, student, predictor, mask_generators,
                       raw_batch: torch.Tensor, validation_keys: list[str],
                       fixed_masks: list[tuple[torch.Tensor, torch.Tensor]],
                       validation_mask_seeds: list[int],
                       dtype: torch.dtype, device, lpips_model):
    if raw_batch.shape[0] == 1:
        raw = raw_batch.to(device)
        full_grid, token_types, _ = build_full_jepa_grid(
            raw, cfg, mask_generators, student, predictor, device, fixed_masks=fixed_masks[0])
        coarse_latent = stage1_decoder(full_grid, token_types)
        target_latent = encode_video_latents(pipe, raw, dtype)
        return run_validation(
            step, output_dir, stage2_cfg, cfg, pipe, refiner, raw,
            full_grid, token_types, coarse_latent, target_latent,
            validation_keys[0] if validation_keys else "unknown", dtype,
            validation_mask_seeds[0], lpips_model)

    refiner.eval()
    val_dir = output_dir / "validation" / f"step_{step:07d}"
    fps = stage2_cfg["validation"]["fps"]
    paths = []
    records = []
    for idx, key in enumerate(validation_keys):
        raw = raw_batch[idx:idx + 1].to(device)
        full_grid, token_types, grid_meta = build_full_jepa_grid(
            raw, cfg, mask_generators, student, predictor, device,
            fixed_masks=fixed_masks[idx])
        coarse_latent = stage1_decoder(full_grid, token_types)
        target_latent = encode_video_latents(pipe, raw, dtype)
        refined_latent = refiner(coarse_latent, full_grid, token_types)
        _, refined_metrics = compute_loss(
            pred_latent=refined_latent,
            target_latent=target_latent,
            raw_batch=raw,
            pipe=pipe,
            dtype=dtype,
            num_frames=cfg["data"]["num_frames"],
            loss_cfg=stage2_cfg["loss"],
            lpips_model=lpips_model,
            step=0,
        )
        coarse_mse = F.mse_loss(coarse_latent.float(), target_latent.float()).item()
        coarse_l1 = F.l1_loss(coarse_latent.float(), target_latent.float()).item()
        record = {
            "validation_index": idx,
            "clip_key": key,
            "validation_mask_base_seed": stage2_cfg["validation"]["seed"],
            "validation_mask_seed": validation_mask_seeds[idx],
            **refined_metrics,
            "stage1_coarse_latent_mse": float(coarse_mse),
            "stage1_coarse_latent_l1": float(coarse_l1),
            "latent_mse_improvement": float(coarse_mse - refined_metrics["loss_latent_mse"]),
        }
        outputs = {
            f"val_{idx:02d}_reference_input": raw_batch_to_frames(raw.detach().cpu()),
            f"val_{idx:02d}_stage1_coarse": decode_latents_to_frames(pipe, coarse_latent, dtype, cfg["data"]["num_frames"]),
            f"val_{idx:02d}_stage2_refined": decode_latents_to_frames(pipe, refined_latent, dtype, cfg["data"]["num_frames"]),
            f"val_{idx:02d}_cosmos_vae_roundtrip": decode_latents_to_frames(pipe, target_latent, dtype, cfg["data"]["num_frames"]),
        }
        for name, frames in outputs.items():
            path = val_dir / f"{name}.mp4"
            export_video(frames, path, fps)
            paths.append(path)
        meta_path = val_dir / f"val_{idx:02d}.json"
        meta_path.write_text(json.dumps({
            "step": step,
            "validation_index": idx,
            "clip_key": key,
            "validation_mask_base_seed": stage2_cfg["validation"]["seed"],
            "validation_mask_seed": validation_mask_seeds[idx],
            "jepa_grid": grid_meta,
            "coarse_latent_shape": list(coarse_latent.shape),
            "target_latent_shape": list(target_latent.shape),
            "refined_latent_shape": list(refined_latent.shape),
            "metrics": record,
        }, indent=2) + "\n")
        paths.append(meta_path)
        records.append(record)
    summary = build_validation_summary(step, records)
    metrics_path = val_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    paths.append(metrics_path)
    refiner.train()
    print(f"\n[validation] saved {len(validation_keys)} Stage-2 validation samples under {val_dir}")
    return paths, summary


def train(cfg: dict, stage2_cfg: dict, args):
    check_gpu()
    print_cgroup_header(prefix="[stage2-latent-refiner]")
    start_oom_watchdog(prefix="[stage2-latent-refiner]-oom-watchdog")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(cfg["data"]["seed"])

    output_dir = Path(args.output_dir)
    wipe_output_dir(output_dir, args.cache_policy, label=f"output_dir ({output_dir.name})")
    output_dir.mkdir(parents=True, exist_ok=True)

    indexed_split_cfg = stage2_cfg.get("indexed_split", {})
    random_split_cfg = stage2_cfg.get("random_split", {})
    indexed_split_enabled = bool(indexed_split_cfg.get("enabled", False))
    random_split_enabled = bool(random_split_cfg.get("enabled", False))
    if indexed_split_enabled and random_split_enabled:
        raise ValueError("enable only one of indexed_split or random_split")
    split_cfg = random_split_cfg if random_split_enabled else indexed_split_cfg
    multi_sample_mode = bool(split_cfg.get("enabled", False)) and not args.input_mp4 and not args.clip_key
    if args.max_tar_files is None:
        args.max_tar_files = (
            split_cfg.get("max_tar_files")
            if multi_sample_mode else stage2_cfg["overfit"]["max_tar_files"]
        )
    if not args.input_mp4:
        args.local_data = ensure_local_data(args)

    source_ckpt = resolve_source_ckpt(args)
    student, predictor = load_frozen_jepa(
        Path(source_ckpt), cfg["model"], cfg["data"], device, dtype=dtype)
    mask_generators = build_mask_generators(cfg)
    validation_seed = stage2_cfg["validation"].get(
        "seed", stage2_cfg["overfit"].get("fixed_mask_seed", 1))
    fixed_masks = sample_fixed_validation_masks(
        mask_generators, batch_size=1, seed=validation_seed)

    stage1_decoder, stage1_payload = load_stage1_decoder(Path(args.stage1_ckpt), device)
    stage1_decoder.requires_grad_(False)
    stage1_decoder.eval()

    stage1_cfg = stage1_payload.get("stage_cfg", {})
    pipe = load_cosmos_vae(
        stage1_payload["cosmos_model_id"],
        stage1_payload["cosmos_revision"],
        dtype,
        device,
        vae_subfolder=stage1_cfg.get("cosmos_vae_subfolder", "vae"),
        enable_tiling=stage1_cfg.get("cosmos_vae_enable_tiling", False),
        enable_slicing=stage1_cfg.get("cosmos_vae_enable_slicing", False),
    )
    if stage2_cfg["memory"]["vae_gradient_checkpointing"]:
        supports_gc = bool(getattr(pipe.vae, "_supports_gradient_checkpointing", False))
        if supports_gc and hasattr(pipe.vae, "enable_gradient_checkpointing"):
            try:
                pipe.vae.enable_gradient_checkpointing()
                print("Enabled gradient checkpointing for decoded Stage-2 VAE losses")
            except (AttributeError, NotImplementedError, ValueError) as exc:
                print(f"  WARN: Cosmos VAE gradient checkpointing unavailable; continuing without it: {exc}")
        else:
            print("  WARN: AutoencoderKLWan does not support gradient checkpointing; "
                  "continuing without it")
    lpips_model = build_lpips_model(stage2_cfg["loss"], device)
    wandb_module = init_wandb(stage2_cfg, cfg, args, output_dir)
    print("Stage-2 loss weights: "
          f"latent_mse={stage2_cfg['loss']['latent_mse_weight']} "
          f"latent_l1={stage2_cfg['loss']['latent_l1_weight']} "
          f"rgb_l1={stage2_cfg['loss']['rgb_l1_weight']} "
          f"spatial_grad={stage2_cfg['loss']['spatial_gradient_weight']} "
          f"temporal_diff={stage2_cfg['loss']['temporal_difference_weight']} "
          f"lpips={stage2_cfg['loss']['lpips_weight']}")

    prod = None
    stop_event = None
    train_q = None
    validation_raw = None
    validation_keys = None
    split_meta = None
    first_batch = None

    def next_training_batch():
        while True:
            try:
                msg_type, raw, keys = train_q.get(timeout=600)
            except queue.Empty as exc:
                raise RuntimeError("Stage-2 producer timeout") from exc
            if msg_type == "error":
                raise RuntimeError("Stage-2 producer failed")
            if msg_type == "done":
                continue
            return raw.to(device), keys

    if multi_sample_mode:
        if random_split_enabled:
            train_keys, val_keys, test_keys, split_meta, manifest_path = load_or_create_random_split(
                args.local_data,
                split_cfg,
                manifest_path=args.split_manifest,
                require_existing=True,
            )
            expected_manifest_hash = (
                stage1_payload.get("stage_cfg", {})
                .get("random_split", {})
                .get("manifest_sha256")
            )
            if not expected_manifest_hash:
                raise RuntimeError(
                    "Stage-1 checkpoint does not record a random split manifest hash. "
                    "Train Stage-1 with the shared random-split pipeline before Stage-2."
                )
            if expected_manifest_hash != split_meta["manifest_sha256"]:
                raise RuntimeError(
                    "Stage-1 checkpoint and Stage-2 split manifest differ: "
                    f"stage1={expected_manifest_hash}, current={split_meta['manifest_sha256']}"
                )
            split_cfg["manifest_path"] = str(manifest_path)
            split_cfg["manifest_sha256"] = split_meta["manifest_sha256"]
            output_manifest_path = output_dir / "dataset_split_manifest.json"
            shutil.copy2(manifest_path, output_manifest_path)
            wandb_save_file(wandb_module, output_manifest_path, output_dir)
            wandb_log_artifact(
                wandb_module,
                output_manifest_path,
                "stage2-dataset-split",
                "dataset-split",
                aliases=["shared-split"],
            )
        else:
            train_keys, val_keys, split_meta = build_indexed_split(args.local_data, split_cfg)
            test_keys = []
        validation_raw, validation_keys = load_validation_set(
            args.local_data, val_keys, cfg, max_tar_files=args.max_tar_files)
        if not random_split_enabled:
            split_path = output_dir / "validation_split.json"
            split_path.write_text(json.dumps({
                **split_meta,
                "train_clip_keys": sorted(train_keys),
                "validation_clip_keys": validation_keys,
            }, indent=2) + "\n")
        if random_split_enabled:
            val_every, steps_per_epoch = resolve_validation_every_n_steps(
                stage2_cfg["validation"], len(train_keys), cfg["optimization"]["batch_size"])
            stage2_cfg["validation"]["every_n_steps"] = val_every
            stage2_cfg["validation"]["steps_per_epoch"] = steps_per_epoch
            print(
                f"Epoch schedule: {steps_per_epoch} optimizer steps/epoch; "
                f"validation every {val_every} steps "
                f"({stage2_cfg['validation']['every_n_epochs']} epoch)"
            )
            if stage2_cfg["checkpoint_on_validation"]:
                print(f"Checkpoint schedule: every validation ({val_every} steps)")
        train_q = queue.Queue(maxsize=get_pipeline_config()["streaming"]["prefetch_queue_train"])
        stop_event = threading.Event()
        prod = threading.Thread(
            target=producer_thread,
            args=(cfg, train_q, stop_event, train_keys, args.local_data, args.max_tar_files),
            daemon=True,
        )
        prod.start()
        raw_batch, batch_keys = next_training_batch()
    else:
        clip_key, raw_batch = load_input_sample(args, cfg, args.max_tar_files)
        batch_keys = [clip_key]
        raw_batch = raw_batch.to(device)
        validation_raw = raw_batch.detach().cpu()
        validation_keys = batch_keys

    if multi_sample_mode:
        validation_masks, validation_mask_seeds = build_per_clip_validation_masks(
            mask_generators, validation_keys, validation_seed)
        print(
            f"Deterministic per-clip validation masks: {len(validation_masks)} clips, "
            f"base seed={validation_seed}"
        )
    else:
        validation_masks, validation_mask_seeds = build_per_clip_validation_masks(
            mask_generators,
            validation_keys,
            validation_seed,
            shared_masks=fixed_masks,
            shared_seed=validation_seed,
        )

    with torch.no_grad():
        full_grid, token_types, grid_meta = build_full_jepa_grid(
            raw_batch, cfg, mask_generators, student, predictor, device,
            fixed_masks=None if multi_sample_mode else fixed_masks)
        coarse_latent = stage1_decoder(full_grid, token_types)
        target_latent = encode_video_latents(pipe, raw_batch, dtype)
    first_batch = (raw_batch, batch_keys, full_grid, token_types, grid_meta, coarse_latent, target_latent)

    if coarse_latent.shape != target_latent.shape:
        raise RuntimeError(f"Stage1 latent {tuple(coarse_latent.shape)} != target latent {tuple(target_latent.shape)}")

    latent_shape = tuple(target_latent.shape[1:])
    refiner = Stage2LatentRefiner(
        latent_channels=latent_shape[0],
        embed_dim=cfg["model"]["embed_dim"],
        n_levels=cfg["model"]["n_output_distillation"],
        hidden_dim=stage2_cfg["hidden_dim"],
        depth=stage2_cfg["depth"],
        kernel_size=stage2_cfg["kernel_size"],
        dropout=stage2_cfg["dropout"],
        cond_dim=stage2_cfg["cond_dim"],
        residual_scale=stage2_cfg["residual_scale"],
    ).to(device)
    print_cuda_memory("Stage-2 refiner initialization")

    optimizer = torch.optim.AdamW(
        refiner.parameters(), lr=stage2_cfg["learning_rate"], weight_decay=stage2_cfg["weight_decay"])
    from diffusers.optimization import get_linear_schedule_with_warmup
    total_steps = int(
        split_cfg.get("num_training_steps", stage2_cfg["overfit"]["num_training_steps"])
        if multi_sample_mode else stage2_cfg["overfit"]["num_training_steps"]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=stage2_cfg["scheduler_warmup_steps"],
        num_training_steps=total_steps,
    )

    ckpt_path = output_dir / CHECKPOINT_LATEST
    start_step = 0
    if ckpt_path.exists():
        payload = torch_load_cpu(ckpt_path, weights_only=False)
        if random_split_enabled and multi_sample_mode:
            resume_manifest_hash = (
                payload.get("stage2_cfg", {})
                .get("random_split", {})
                .get("manifest_sha256")
            )
            if resume_manifest_hash != split_meta["manifest_sha256"]:
                raise RuntimeError(
                    "latest Stage-2 checkpoint was trained with a different split manifest: "
                    f"checkpoint={resume_manifest_hash}, current={split_meta['manifest_sha256']}"
                )
        refiner.load_state_dict(payload["refiner_state_dict"])
        if "optimizer" in payload and "scheduler" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
        else:
            print("  WARN: latest checkpoint has no optimizer/scheduler state; resuming weights with fresh optimizer")
        start_step = int(payload["step"])
        del payload
        print(f"Resumed Stage-2 refiner from step {start_step}")

    log_path = output_dir / "loss_log.jsonl"
    log_file = log_path.open("a")
    pbar = make_pbar(total=total_steps, initial=start_step, desc="stage2_latent_refiner", unit="step")
    n_trainable = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    mode_name = "random split" if random_split_enabled and multi_sample_mode else (
        "indexed split" if multi_sample_mode else "overfit")
    print(f"\n=== Stage-2 latent refiner {mode_name}: {start_step} -> {total_steps} steps ===")
    if multi_sample_mode:
        if random_split_enabled:
            print(f"TARs: {split_meta['tar_files']}")
            print(f"Training clips: {split_meta['train_num_samples']}")
            print(f"Validation clips: {split_meta['validation_num_samples']}")
            print(f"Held-out test clips: {split_meta['test_num_samples']}")
            print(f"Shared split manifest: {split_meta['manifest_path']}")
        else:
            print(f"TAR: {split_meta['tar_path']}")
            print(f"Training clips: {split_meta['train_num_samples']}")
            print(f"Validation clips: {split_meta['validation_num_samples']}")
            print(f"Validation one-based indices: {split_meta['validation_indices_one_based']}")
    else:
        print(f"Fixed clip: {batch_keys[0]}")
    print(f"Fixed validation JEPA mask seed: {validation_seed}")
    print(f"Latent shape: {latent_shape}; trainable params: {n_trainable / 1e6:.2f}M")
    print(f"JEPA grid: {grid_meta}")

    try:
        if stage2_cfg["validation"]["enabled"] and stage2_cfg["validation"]["run_before_training"] and start_step == 0:
            ref_paths = save_validation_references(
                validation_raw, validation_keys, validation_masks,
                validation_mask_seeds, cfg, stage2_cfg, output_dir)
            wandb_log_videos(
                wandb_module, ref_paths, 0,
                "validation/reference_input", stage2_cfg["validation"]["fps"])
            wandb_log_artifact_many(
                wandb_module, ref_paths,
                "stage2-validation-reference", "validation-reference")
            val_paths, val_summary = run_validation_set(
                0, output_dir, stage2_cfg, cfg, pipe, refiner, stage1_decoder,
                student, predictor, mask_generators, validation_raw, validation_keys,
                validation_masks, validation_mask_seeds, dtype, device, lpips_model)
            wandb_log_videos(
                wandb_module, val_paths, 0,
                "validation/stage2/videos", stage2_cfg["validation"]["fps"])
            wandb_log_validation_summary(
                wandb_module, val_summary, 0, "validation/stage2")
            wandb_log_artifact_many(
                wandb_module, val_paths,
                "stage2-validation-step-0000000", "validation",
                aliases=["step-0000000"])
            cuda_cleanup()

        for step in range(start_step, total_steps):
            if multi_sample_mode:
                if first_batch is not None:
                    raw_batch, batch_keys, full_grid, token_types, grid_meta, coarse_latent, target_latent = first_batch
                    first_batch = None
                else:
                    raw_batch, batch_keys = next_training_batch()
                    with torch.no_grad():
                        full_grid, token_types, grid_meta = build_full_jepa_grid(
                            raw_batch, cfg, mask_generators, student, predictor, device, fixed_masks=None)
                        coarse_latent = stage1_decoder(full_grid, token_types)
                        target_latent = encode_video_latents(pipe, raw_batch, dtype)
            refiner.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                pred_latent = refiner(coarse_latent.detach(), full_grid.detach(), token_types)
                loss, loss_terms = compute_loss(
                    pred_latent=pred_latent,
                    target_latent=target_latent,
                    raw_batch=raw_batch,
                    pipe=pipe,
                    dtype=dtype,
                    num_frames=cfg["data"]["num_frames"],
                    loss_cfg=stage2_cfg["loss"],
                    lpips_model=lpips_model,
                    step=step,
                )
                if loss.grad_fn is None:
                    n_trainable = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
                    raise RuntimeError(
                        "Stage-2 loss has no grad_fn; the refiner forward is not connected to autograd "
                        f"(grad_enabled={torch.is_grad_enabled()}, "
                        f"pred_requires_grad={pred_latent.requires_grad}, "
                        f"trainable_params={n_trainable})"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), stage2_cfg["grad_clip"])
                optimizer.step()
            scheduler.step()

            lr_val = scheduler.get_last_lr()[0]
            record = {"step": step, "lr": lr_val, **loss_terms}
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            os.fsync(log_file.fileno())
            wandb_log_metrics(wandb_module, record)
            pbar.set_postfix_str(
                f"loss={loss.item():.5f} "
                f"mse={loss_terms['loss_latent_mse']:.5f} "
                f"rgb={loss_terms['loss_rgb_l1']:.5f} "
                f"lpips={loss_terms['loss_lpips']:.5f} "
                f"lr={lr_val:.2e}"
            )
            pbar.update(1)

            if periodic_checkpoint_due(stage2_cfg, step + 1):
                step_path = step_checkpoint_path(output_dir, step + 1)
                step_saved = save_checkpoint(
                    step_path, refiner, optimizer, scheduler, step + 1, stage2_cfg,
                    args.stage1_ckpt, source_ckpt, cfg["model"], cfg["data"], latent_shape,
                    include_optimizer=stage2_cfg["checkpoint_include_optimizer_for_step_files"])
                latest_saved = save_checkpoint(
                    ckpt_path, refiner, optimizer, scheduler, step + 1, stage2_cfg,
                    args.stage1_ckpt, source_ckpt, cfg["model"], cfg["data"], latent_shape,
                    include_optimizer=True)
                if step_saved:
                    wandb_save_file(wandb_module, step_path, output_dir)
                    wandb_log_artifact(
                        wandb_module, step_path,
                        f"stage2-latent-refiner-step-{step + 1:07d}", "model",
                        aliases=[f"step-{step + 1:07d}"])
                if latest_saved:
                    wandb_save_file(wandb_module, ckpt_path, output_dir)
                    wandb_log_artifact(
                        wandb_module, ckpt_path,
                        "stage2-latent-refiner-latest", "model",
                        aliases=["latest"])

            val_every = stage2_cfg["validation"]["every_n_steps"]
            if stage2_cfg["validation"]["enabled"] and val_every > 0 and (step + 1) % val_every == 0:
                val_paths, val_summary = run_validation_set(
                    step + 1, output_dir, stage2_cfg, cfg, pipe, refiner, stage1_decoder,
                    student, predictor, mask_generators, validation_raw, validation_keys,
                    validation_masks, validation_mask_seeds, dtype, device, lpips_model)
                wandb_log_videos(
                    wandb_module, val_paths, step + 1,
                    "validation/stage2/videos", stage2_cfg["validation"]["fps"])
                wandb_log_validation_summary(
                    wandb_module, val_summary, step + 1, "validation/stage2")
                wandb_log_artifact_many(
                    wandb_module, val_paths,
                    f"stage2-validation-step-{step + 1:07d}", "validation",
                    aliases=[f"step-{step + 1:07d}"])
                cuda_cleanup()
    finally:
        pbar.close()
        log_file.close()
        if stop_event is not None:
            stop_event.set()
        if prod is not None:
            prod.join(timeout=5)
        final_step = step + 1 if "step" in locals() else start_step
        if final_step > 0:
            final_step_path = step_checkpoint_path(output_dir, final_step)
            if not final_step_path.exists():
                save_checkpoint(
                    final_step_path, refiner, optimizer, scheduler, final_step, stage2_cfg,
                    args.stage1_ckpt, source_ckpt, cfg["model"], cfg["data"], latent_shape,
                    include_optimizer=stage2_cfg["checkpoint_include_optimizer_for_step_files"])
            save_checkpoint(
                ckpt_path, refiner, optimizer, scheduler, final_step, stage2_cfg,
                args.stage1_ckpt, source_ckpt, cfg["model"], cfg["data"], latent_shape,
                include_optimizer=True)

    final_path = output_dir / CHECKPOINT_FINAL
    final_saved = save_checkpoint(
        final_path, refiner, optimizer, scheduler, final_step, stage2_cfg,
        args.stage1_ckpt, source_ckpt, cfg["model"], cfg["data"], latent_shape,
        include_optimizer=stage2_cfg["checkpoint_include_optimizer_for_final"])
    final_step_path = step_checkpoint_path(output_dir, final_step)
    if final_saved:
        wandb_save_file(wandb_module, final_path, output_dir)
        wandb_log_artifact(wandb_module, final_path, "stage2-latent-refiner-final", "model", aliases=["final"])
    if final_step_path.exists():
        wandb_save_file(wandb_module, final_step_path, output_dir)
        wandb_log_artifact(
            wandb_module, final_step_path,
            f"stage2-latent-refiner-step-{final_step:07d}", "model",
            aliases=[f"step-{final_step:07d}"])
    if ckpt_path.exists():
        wandb_save_file(wandb_module, ckpt_path, output_dir)
        wandb_log_artifact(wandb_module, ckpt_path, "stage2-latent-refiner-latest", "model", aliases=["latest"])
    wandb_save_file(wandb_module, log_path, output_dir)
    wandb_log_artifact(wandb_module, log_path, "stage2-loss-log", "loss-log", aliases=["latest"])
    if wandb_module is not None:
        wandb_module.finish()
    print(f"\nSaved Stage-2 refiner: {final_path}")


def main():
    parser = argparse.ArgumentParser("Train Stage-2 latent refiner over a frozen Stage-1 decoder")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--stage2-config", default="configs/stage2_latent_refiner.yaml")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--source-ckpt", default=DEFAULT_SOURCE_CKPT)
    parser.add_argument("--source-hf-repo", default=DEFAULT_SOURCE_HF_REPO)
    parser.add_argument("--source-hf-repo-type", default="dataset")
    parser.add_argument("--source-hf-filename", default=DEFAULT_SOURCE_HF_FILENAME)
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
        help="Shared split manifest created by Stage-1. Required for matching random splits.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--input-mp4", default=None)
    parser.add_argument("--clip-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overfit-steps", type=int, default=None)
    parser.add_argument("--overfit-mask-seed", type=int, default=None)
    parser.add_argument("--validation-every-n-steps", type=int, default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-api-key", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    add_cache_policy_arg(parser)
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        stage1_train.HF_TOKEN = args.hf_token

    args.hf_dataset_repo = args.dataset_id

    args.SANITY = False
    args.POC = False
    args.FULL = True
    args.subset = None
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    cfg = load_merged_config(args.model_config, args.train_config)
    stage2_cfg = load_config(args.stage2_config)
    stage2_cfg.setdefault("loss", {})
    stage2_cfg["loss"].setdefault("latent_mse_weight", 1.0)
    stage2_cfg["loss"].setdefault("latent_l1_weight", 0.1)
    stage2_cfg["loss"].setdefault("rgb_l1_weight", 0.2)
    stage2_cfg["loss"].setdefault("spatial_gradient_weight", 0.05)
    stage2_cfg["loss"].setdefault("temporal_difference_weight", 0.05)
    stage2_cfg["loss"].setdefault("lpips_weight", 0.05)
    stage2_cfg["loss"].setdefault("lpips_net", "alex")
    stage2_cfg["loss"].setdefault("lpips_max_frames", 4)
    stage2_cfg["loss"].setdefault("charbonnier_eps", 1.0e-3)
    stage2_cfg["loss"].setdefault("decoded_loss_every_n_steps", 1)
    stage2_cfg.setdefault("checkpoint_every_n_steps", 50)
    stage2_cfg.setdefault("checkpoint_on_validation", False)
    stage2_cfg.setdefault("checkpoint_include_optimizer_for_step_files", False)
    stage2_cfg.setdefault("checkpoint_include_optimizer_for_final", False)
    stage2_cfg.setdefault("memory", {})
    stage2_cfg["memory"].setdefault("vae_gradient_checkpointing", True)
    stage2_cfg.setdefault("overfit", {})
    stage2_cfg["overfit"].setdefault("enabled", True)
    stage2_cfg["overfit"].setdefault("num_training_steps", 1000)
    stage2_cfg["overfit"].setdefault("fixed_mask_seed", 1)
    stage2_cfg["overfit"].setdefault("max_tar_files", 1)
    stage2_cfg.setdefault("indexed_split", {})
    stage2_cfg["indexed_split"].setdefault("enabled", False)
    stage2_cfg["indexed_split"].setdefault("tar_index", 0)
    stage2_cfg["indexed_split"].setdefault("source_num_samples", 1000)
    stage2_cfg["indexed_split"].setdefault("validation_every_nth", 50)
    stage2_cfg["indexed_split"].setdefault("one_based_nth", True)
    stage2_cfg["indexed_split"].setdefault("num_training_steps", 10000)
    stage2_cfg["indexed_split"].setdefault("max_tar_files", stage2_cfg["indexed_split"]["tar_index"] + 1)
    stage2_cfg.setdefault("random_split", {})
    stage2_cfg["random_split"].setdefault("enabled", False)
    stage2_cfg["random_split"].setdefault("manifest_path", "data/splits/stage12_split.json")
    stage2_cfg["random_split"].setdefault("seed", 42)
    stage2_cfg["random_split"].setdefault("max_tar_files", 3)
    stage2_cfg["random_split"].setdefault("source_num_samples", 3000)
    stage2_cfg["random_split"].setdefault("test_fraction", 0.10)
    stage2_cfg["random_split"].setdefault("validation_num_samples", 50)
    stage2_cfg["random_split"].setdefault("num_training_steps", 10000)
    stage2_cfg.setdefault("validation", {})
    stage2_cfg["validation"].setdefault("enabled", True)
    stage2_cfg["validation"].setdefault("run_before_training", True)
    stage2_cfg["validation"].setdefault("every_n_steps", stage2_cfg["checkpoint_every_n_steps"])
    stage2_cfg["validation"].setdefault("every_n_epochs", None)
    stage2_cfg["validation"].setdefault("seed", 1)
    stage2_cfg["validation"].setdefault("fps", 16)
    stage2_cfg.setdefault("wandb", {})
    stage2_cfg["wandb"].setdefault("enabled", False)
    stage2_cfg["wandb"].setdefault("project", "factorjepa-stage2")
    stage2_cfg["wandb"].setdefault("entity", "")
    stage2_cfg["wandb"].setdefault("run_name", "")
    stage2_cfg["wandb"].setdefault("group", "")
    stage2_cfg["wandb"].setdefault("tags", ["stage2", "latent-refiner", "jepa-decoder", "cosmos-vae"])
    stage2_cfg["wandb"].setdefault("resume", "allow")
    stage2_cfg["wandb"].setdefault("api_key", "")
    if args.overfit_steps is not None:
        stage2_cfg["overfit"]["num_training_steps"] = args.overfit_steps
    if args.overfit_mask_seed is not None:
        stage2_cfg["overfit"]["fixed_mask_seed"] = args.overfit_mask_seed
    if args.validation_every_n_steps is not None:
        stage2_cfg["validation"]["every_n_steps"] = args.validation_every_n_steps
    if args.wandb:
        stage2_cfg["wandb"]["enabled"] = True
    if args.wandb_api_key is not None:
        stage2_cfg["wandb"]["api_key"] = args.wandb_api_key
    if args.wandb_project is not None:
        stage2_cfg["wandb"]["project"] = args.wandb_project
    if args.wandb_entity is not None:
        stage2_cfg["wandb"]["entity"] = args.wandb_entity
    if args.wandb_run_name is not None:
        stage2_cfg["wandb"]["run_name"] = args.wandb_run_name
    if args.wandb_group is not None:
        stage2_cfg["wandb"]["group"] = args.wandb_group
    if args.batch_size is not None:
        cfg["optimization"]["batch_size"] = args.batch_size
    split_for_data = (
        stage2_cfg["random_split"]
        if stage2_cfg["random_split"]["enabled"] else stage2_cfg["indexed_split"]
    )
    random_multi_sample = (
        stage2_cfg["random_split"]["enabled"] and not args.input_mp4 and not args.clip_key
    )
    if random_multi_sample:
        expected_max_tars = int(stage2_cfg["random_split"]["max_tar_files"])
        if args.max_tar_files is not None and args.max_tar_files != expected_max_tars:
            raise ValueError(
                f"--max-tar-files={args.max_tar_files} conflicts with shared random split "
                f"max_tar_files={expected_max_tars}"
            )
        args.max_tar_files = expected_max_tars
    if args.max_tar_files is None:
        args.max_tar_files = (
            split_for_data["max_tar_files"]
            if split_for_data["enabled"] and not args.input_mp4 and not args.clip_key
            else stage2_cfg["overfit"]["max_tar_files"]
        )
    if stage2_cfg["validation"]["every_n_steps"] is None and not random_multi_sample:
        stage2_cfg["validation"]["every_n_steps"] = stage2_cfg["checkpoint_every_n_steps"]

    train(cfg, stage2_cfg, args)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback
        print(f"\nFATAL (stage2-latent-refiner): {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
