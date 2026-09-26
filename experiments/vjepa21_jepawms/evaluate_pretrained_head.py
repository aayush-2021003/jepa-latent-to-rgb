"""Zero-shot JEPA-WMs decoder readout of cached V-JEPA 2.1 frame-15 features.

This is an evaluation-only diagnostic, not an adapted-head training baseline.
It uses the same source-disjoint clips and cached features as the 10K Cosmos run.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.vjepa21_cosmos_single_frame.common import (
    atomic_json_dump,
    load_config as load_cosmos_config,
    project_path,
    require_cuda,
    seed_everything,
)
from experiments.vjepa21_cosmos_single_frame.data import LatentShardDataset, cache_collate
from experiments.vjepa21_jepawms.common import load_config as load_head_config
from experiments.vjepa21_jepawms.models import FrozenJEPAWMSImageDecoder
from experiments.jepa_cosmos.tracking import log_file_artifact, start_wandb


METRICS = ("l1", "mse", "psnr", "lpips")
SOURCES = ("predicted", "oracle", "persistence")


def decoder_features(cached: torch.Tensor, grid: int, channels: int) -> torch.Tensor:
    """Convert a cached one-tubelet volume to the pretrained head's input grid."""
    if cached.ndim != 5 or cached.shape[1] != channels or cached.shape[2] != 1:
        raise ValueError(
            f"Expected (B,{channels},1,H,W) cached features, got {tuple(cached.shape)}"
        )
    frame = cached[:, :, 0].float()
    if frame.shape[-2:] != (grid, grid):
        frame = F.interpolate(frame, size=(grid, grid), mode="bilinear", align_corners=False)
    return frame.unsqueeze(1)  # (B,1,C,H,W)


def resize_rgb(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if image.shape[-2:] == size:
        return image.float()
    return F.interpolate(image.float(), size=size, mode="bilinear", align_corners=False)


def score(prediction: torch.Tensor, target: torch.Tensor, perceptual) -> dict[str, torch.Tensor]:
    delta = prediction.float() - target.float()
    mse = delta.square().flatten(1).mean(1)
    return {
        "l1": delta.abs().flatten(1).mean(1),
        "mse": mse,
        "psnr": -10.0 * torch.log10(mse.clamp_min(1.0e-10)),
        "lpips": perceptual(prediction.float() * 2 - 1, target.float() * 2 - 1)
        .flatten(1).mean(1),
    }


def save_panel(path: Path, key: str, images: dict[str, torch.Tensor]) -> None:
    headers = (
        ("Context frame 14", "context"),
        ("GT frame 15", "ground_truth"),
        ("JEPA-WMs / true JEPA", "oracle"),
        ("JEPA-WMs / predicted JEPA", "predicted"),
    )
    panels = []
    for _, name in headers:
        value = images[name].detach().cpu().float().clamp(0, 1)
        pixels = (value * 255).round().byte().permute(1, 2, 0).numpy()
        panels.append(Image.fromarray(pixels, mode="RGB"))
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height + 58), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, ((header, _), panel) in enumerate(zip(headers, panels)):
        left = index * width
        canvas.paste(panel, (left, 30))
        draw.text((left + 6, 9), header, fill="black", font=font)
    draw.text((6, height + 37), f"frames 1-14 observed; frame 15 held out | {key}", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def validate_cache(dataset: LatentShardDataset, split: str, expected: int) -> None:
    metadata = dataset.manifest.get("metadata", {})
    required = {
        "split": split,
        "context_frames": 14,
        "target_frame_number": 15,
        "contains_jepa_predicted": True,
        "contains_jepa_target": True,
        "contains_target_rgb": True,
        "contains_previous_rgb": True,
        "input_dim": 1408,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in required.items() if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Cache is not the expected frame-15 evaluation set: {mismatches}")
    actual = int(dataset.manifest["samples"])
    if actual != expected:
        raise RuntimeError(f"{split} cache has {actual} samples; expected {expected}")


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    cosmos_config = load_cosmos_config(args.config)
    head_config = load_head_config(args.head_config)
    seed_everything(int(cosmos_config["experiment"]["seed"]))
    device = require_cuda()
    cache_root = project_path(cosmos_config["data"]["cache_root"])
    dataset = LatentShardDataset(
        cache_root / args.split, shuffle=False,
        seed=int(cosmos_config["experiment"]["seed"]), max_samples=args.max_samples,
    )
    validate_cache(dataset, args.split, int(cosmos_config["data"][f"{args.split}_samples"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0,
                        pin_memory=True, collate_fn=cache_collate)
    decoder = FrozenJEPAWMSImageDecoder(head_config, device)
    channels = int(head_config["jepa_wms_decoder"]["architecture"]["embed_dim"])
    if channels != int(cosmos_config["vjepa21"]["embed_dim"]):
        raise RuntimeError("V-JEPA cache and JEPA-WMs decoder feature widths differ")
    if args.feature_grid <= 0:
        raise ValueError("--feature-grid must be positive")
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("LPIPS is required for matched Cosmos metrics") from error
    backbone = cosmos_config["loss"].get("perceptual_backbone", "alex")
    perceptual = lpips.LPIPS(net=backbone).to(device).eval().requires_grad_(False)
    output = project_path(args.output_dir) / args.split
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    media_paths: list[Path] = []
    native_sizes: set[tuple[int, int]] = set()
    for batch in tqdm(loader, desc=f"JEPA-WMs head {args.split}"):
        target = batch["target_rgb"].to(device).float() / 255.0
        context = batch["previous_rgb"].to(device).float() / 255.0
        image_size = tuple(target.shape[-2:])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = decoder.decode_rgb01(decoder_features(
                batch["jepa_predicted"].to(device), args.feature_grid, channels,
            ))[:, 0]
            oracle = decoder.decode_rgb01(decoder_features(
                batch["jepa_target"].to(device), args.feature_grid, channels,
            ))[:, 0]
        native_sizes.add(tuple(prediction.shape[-2:]))
        if prediction.shape[-2:] != oracle.shape[-2:]:
            raise RuntimeError("Predicted and oracle decoder output sizes differ")
        images = {
            "predicted": resize_rgb(prediction, image_size).clamp(0, 1),
            "oracle": resize_rgb(oracle, image_size).clamp(0, 1),
            "persistence": context,
        }
        scores = {source: score(images[source], target, perceptual) for source in SOURCES}
        for index, key in enumerate(batch["keys"]):
            row = {"key": key}
            for source in SOURCES:
                for metric in METRICS:
                    row[f"{source}_{metric}"] = float(scores[source][metric][index])
            rows.append(row)
            if len(media_paths) < args.media_samples:
                path = output / "panels" / f"sample_{len(media_paths):02d}.png"
                save_panel(path, key, {
                    "context": context[index], "ground_truth": target[index],
                    "oracle": images["oracle"][index],
                    "predicted": images["predicted"][index],
                })
                media_paths.append(path)
    if not rows:
        raise RuntimeError("Evaluation cache yielded zero samples")
    fields = ["key"] + [f"{source}_{metric}" for source in SOURCES for metric in METRICS]
    with (output / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "split": args.split,
        "samples": len(rows),
        "target_frame": 15,
        "context_frames": 14,
        "feature_source": "cached V-JEPA 2.1 final-level predicted/true tubelet",
        "head_checkpoint": str(project_path(head_config["jepa_wms_decoder"]["checkpoint_path"])),
        "head_training_encoder": "V-JEPA-2 ViT-G/16 (not V-JEPA 2.1)",
        "head_feature_grid": args.feature_grid,
        "native_decoder_sizes": sorted([list(size) for size in native_sizes]),
        "metric_image_size": list(image_size),
        "lpips_backbone": backbone,
        "warning": "Zero-shot cross-version readout; not a capacity-matched retrained baseline.",
        "metrics": {
            field: sum(row[field] for row in rows) / len(rows) for field in fields[1:]
        },
    }
    atomic_json_dump(report, output / "summary.json")
    if not args.no_wandb:
        run = start_wandb(
            cosmos_config, "jepa-wms-pretrained-head-evaluation",
            f"vjepa21-jepawms-pretrained-head-frame15-{args.split}",
        )
        run.config.update({
            "baseline/head_config": args.head_config,
            "baseline/feature_grid": args.feature_grid,
            "baseline/zero_shot_cross_version": True,
        })
        payload = {f"{args.split}/{key}": value for key, value in report["metrics"].items()}
        payload[f"{args.split}/samples"] = len(rows)
        if media_paths:
            import wandb
            payload[f"{args.split}/comparisons"] = [wandb.Image(str(path)) for path in media_paths]
        run.log(payload)
        log_file_artifact(run, f"jepawms-pretrained-head-frame15-{args.split}",
                          "evaluation", [output / "summary.json", output / "per_sample.csv", *media_paths])
        run.finish()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="10K Cosmos single-frame config")
    parser.add_argument("--head-config", default="configs/experiments/vjepa21_jepawms_two_frame.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", default="outputs/vjepa21_jepawms_pretrained_head_frame15")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--feature-grid", type=int, default=16,
                        help="V-JEPA-2/256 pretrained head input grid (16x16)")
    parser.add_argument("--media-samples", type=int, default=8)
    parser.add_argument("--max-samples", type=int,
                        help="Small smoke test only; does not change the source split")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.media_samples < 0 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("batch-size and max-samples must be positive; media-samples nonnegative")
    report = evaluate(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
