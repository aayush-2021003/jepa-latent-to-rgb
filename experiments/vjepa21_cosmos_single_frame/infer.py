"""Predict and reconstruct frame 15 from fourteen observed frames."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

from experiments.jepa_cosmos.losses import latent_alignment_loss
from experiments.jepa_cosmos.media import write_video
from experiments.jepa_cosmos.tracking import log_file_artifact, start_wandb
from experiments.vjepa21_cosmos_single_frame.adapter import build_adapter
from experiments.vjepa21_cosmos_single_frame.common import (
    atomic_json_dump,
    imagenet_normalize,
    load_config,
    project_path,
    require_cuda,
    require_secret,
    resize_center_crop_uint8,
)
from experiments.vjepa21_cosmos_single_frame.media import save_comparison, save_rgb
from experiments.vjepa21_cosmos_single_frame.models import (
    CosmosContinuousImageTokenizer,
    OfficialVJEPA21WorldModel,
    validate_model_geometry,
)


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from utils.video_io import decode_video_bytes


def resolve_adapter_checkpoint(config: dict, value: str | None) -> Path:
    if value:
        path = project_path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    local = project_path(config["experiment"]["output_dir"]) / "adapter_best.pt"
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=config["huggingface"]["repo_id"],
            repo_type="model",
            filename="checkpoints/best.pt",
            token=require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",)),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_model_geometry(config)
    device = require_cuda()
    input_path = Path(args.input_video)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    adapter_path = resolve_adapter_checkpoint(config, args.adapter_checkpoint)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_path(config["experiment"]["output_dir"]) / "inference" / input_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = build_adapter(config).to(device).eval()
    state = torch.load(adapter_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(state["adapter"])
    world = OfficialVJEPA21WorldModel(config, device)
    cosmos = CosmosContinuousImageTokenizer(config, device, True, True)
    data = config["data"]
    with tempfile.TemporaryDirectory(prefix="vjepa21_cosmos_ci_infer_") as temporary_dir:
        raw = decode_video_bytes(
            input_path.read_bytes(),
            temporary_dir,
            input_path.name,
            int(data["num_frames"]),
            cache_frames=False,
        )
    if raw is None:
        raise RuntimeError(f"Could not decode {input_path}")
    cropped = resize_center_crop_uint8(raw, int(data["crop_size"]))
    normalized = imagenet_normalize(cropped).unsqueeze(0)
    context_end = int(data["context_frames"])
    target_index = context_end + int(data["target_frame_offset"])
    target_rgb = cropped[target_index]
    target_input = (target_rgb.float() / 127.5 - 1.0).unsqueeze(0).to(device)
    precision = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
        config["training"]["mixed_precision"]
    ]
    with torch.no_grad(), torch.autocast("cuda", dtype=precision):
        predicted_jepa = world.predict_future(normalized[:, :context_end])
        target_jepa = world.target_future(normalized)
        cosmos_target = cosmos.encode(target_input)
        target_shape = tuple(cosmos_target.shape[-2:])
        predicted_latent = adapter(predicted_jepa, target_shape)
        oracle_latent = adapter(target_jepa, target_shape)
    reconstruction = cosmos.decode(cosmos_target)[0]
    oracle = cosmos.decode(oracle_latent)[0]
    predicted = cosmos.decode(predicted_latent)[0]
    _, predicted_parts = latent_alignment_loss(
        predicted_latent,
        cosmos_target,
        config["loss"]["latent_l1"],
        config["loss"]["latent_cosine"],
    )
    context_path = output_dir / "context_frames_01_14.mp4"
    target_path = output_dir / "ground_truth_frame_15.png"
    predicted_path = output_dir / "predicted_frame_15.png"
    comparison_path = output_dir / "comparison_frame_15.png"
    write_video(context_path, cropped[:context_end].permute(1, 0, 2, 3), args.fps)
    save_rgb(target_path, target_rgb)
    save_rgb(predicted_path, predicted)
    save_comparison(
        comparison_path,
        input_path.stem,
        target_rgb,
        reconstruction,
        oracle,
        predicted,
    )
    predicted_01 = (predicted.float().clamp(-1, 1) + 1.0) / 2.0
    target_01 = target_rgb.float().to(device) / 255.0
    mse = (predicted_01 - target_01).square().mean().clamp_min(1.0e-10)
    metrics = {
        "predicted_latent_l1": float(predicted_parts["latent_l1"]),
        "predicted_latent_cosine": float(predicted_parts["latent_cosine"]),
        "predicted_rgb_l1": float((predicted_01 - target_01).abs().mean()),
        "predicted_rgb_psnr": float(-10.0 * torch.log10(mse)),
        "context_frames": 14,
        "predicted_tubelet_frames": [15, 16],
        "reconstructed_frame": 15,
        "input_video": str(input_path),
        "adapter_checkpoint": str(adapter_path),
    }
    metrics_path = output_dir / "metrics.json"
    atomic_json_dump(metrics, metrics_path)
    if not args.no_wandb:
        import wandb

        run = start_wandb(config, "inference", f"infer-{input_path.stem}-frame15")
        run.log(
            {
                **{f"inference/{key}": value for key, value in metrics.items() if isinstance(value, float)},
                "inference/comparison": wandb.Image(str(comparison_path)),
                "inference/predicted_frame15": wandb.Image(str(predicted_path)),
            }
        )
        log_file_artifact(run, f"inference-{input_path.stem}-frame15", "inference", [metrics_path])
        run.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
