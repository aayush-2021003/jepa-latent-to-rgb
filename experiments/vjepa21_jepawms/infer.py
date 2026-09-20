"""Render both held-out future frames from an arbitrary input video."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from experiments.vjepa21_jepawms.common import (
    atomic_json_dump,
    imagenet_normalize,
    load_config,
    project_path,
    require_cuda,
    resize_center_crop_uint8,
    seed_everything,
)
from experiments.vjepa21_jepawms.losses import PerceptualLoss, image_metrics, latent_metrics
from experiments.vjepa21_jepawms.media import save_comparison
from experiments.vjepa21_jepawms.models import (
    FrozenJEPAWMSImageDecoder,
    OfficialVJEPA21WorldModel,
    validate_geometry,
)
from experiments.vjepa21_jepawms.train_adapter import build_adapter
from experiments.jepa_cosmos.tracking import start_wandb


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from utils.video_io import decode_video_bytes


def resize_uint8(frame: torch.Tensor, size: int) -> torch.Tensor:
    value = F.interpolate(
        frame.unsqueeze(0).float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    return value.round().clamp(0, 255).to(torch.uint8)


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu()
    if value.dtype != torch.uint8:
        value = (value.float().clamp(0, 1) * 255).round().to(torch.uint8)
    Image.fromarray(value.permute(1, 2, 0).numpy(), mode="RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", default="outputs/vjepa21_jepawms_inference")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_geometry(config)
    seed_everything(config["experiment"]["seed"])
    device = require_cuda()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    with tempfile.TemporaryDirectory(prefix="vjepa21_infer_") as temporary_dir:
        frames = decode_video_bytes(
            video_path.read_bytes(),
            temporary_dir,
            video_path.name,
            config["data"]["num_frames"],
            False,
        )
    if frames is None:
        raise RuntimeError(f"Could not decode {video_path}")

    data = config["data"]
    cropped = resize_center_crop_uint8(frames, data["crop_size"])
    normalized = imagenet_normalize(cropped).unsqueeze(0)
    last_context = resize_uint8(
        cropped[data["context_frames"] - 1], data["decoder_image_size"]
    )
    target = torch.stack(
        [
            resize_uint8(cropped[index], data["decoder_image_size"])
            for index in data["target_frame_indices"]
        ]
    )

    world_model = OfficialVJEPA21WorldModel(config, device)
    predicted_features = world_model.predict_tubelet(
        normalized[:, : data["context_frames"]]
    )
    target_features = world_model.target_tubelet(normalized)
    del world_model
    torch.cuda.empty_cache()

    checkpoint_path = project_path(
        args.checkpoint
        or str(Path(config["experiment"]["output_dir"]) / "adapter_best.pt")
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Adapter checkpoint not found: {checkpoint_path}")
    adapter = build_adapter(config).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    decoder = FrozenJEPAWMSImageDecoder(config, device)
    perceptual = PerceptualLoss(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = decoder.decode_rgb01(adapter(predicted_features))[0]
        oracle = decoder.decode_rgb01(adapter(target_features))[0]

    target_float = target.to(device).float() / 255.0
    predicted_metrics = image_metrics(prediction, target_float)
    oracle_metrics = image_metrics(oracle, target_float)
    context_metrics = image_metrics(
        last_context.to(device).float().unsqueeze(0) / 255.0, target_float[1:2]
    )
    latent = latent_metrics(predicted_features, target_features)
    metrics = {
        "predicted_l1": predicted_metrics["l1"].mean().item(),
        "predicted_psnr": predicted_metrics["psnr"].mean().item(),
        "predicted_lpips": perceptual(prediction, target_float).item(),
        "oracle_l1": oracle_metrics["l1"].mean().item(),
        "oracle_psnr": oracle_metrics["psnr"].mean().item(),
        "oracle_lpips": perceptual(oracle, target_float).item(),
        "last_frame_l1": context_metrics["l1"].item(),
        "last_frame_psnr": context_metrics["psnr"].item(),
        "latent_l1": latent["l1"].item(),
        "latent_cosine": latent["cosine"].item(),
        "context_frames": data["context_frames"],
        "target_frame_index_zero_based": data["target_frame_index"],
        "checkpoint": str(checkpoint_path),
    }

    save_rgb(output_dir / "last_context.png", last_context)
    for frame_index, frame_number in enumerate((15, 16)):
        save_rgb(output_dir / f"ground_truth_frame{frame_number}.png", target[frame_index])
        save_rgb(output_dir / f"oracle_frame{frame_number}.png", oracle[frame_index])
        save_rgb(output_dir / f"prediction_frame{frame_number}.png", prediction[frame_index])
        metrics[f"frame{frame_number}_predicted_l1"] = predicted_metrics["l1"][frame_index].item()
        metrics[f"frame{frame_number}_predicted_psnr"] = predicted_metrics["psnr"][frame_index].item()
        metrics[f"frame{frame_number}_predicted_lpips"] = perceptual(
            prediction[frame_index : frame_index + 1],
            target_float[frame_index : frame_index + 1],
        ).item()
    comparison = output_dir / "comparison.png"
    save_comparison(
        comparison,
        video_path.name,
        last_context,
        target,
        oracle,
        prediction,
    )
    atomic_json_dump(metrics, output_dir / "metrics.json")

    if args.wandb:
        import wandb

        run = start_wandb(
            config,
            "two-frame-inference",
            f"{config['tracking']['run_name']}-inference",
        )
        run.log({**metrics, "inference/comparison": wandb.Image(str(comparison))})
        run.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
