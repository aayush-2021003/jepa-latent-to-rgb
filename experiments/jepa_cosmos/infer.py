"""Run FactorJEPA context-to-future prediction and decode it through Cosmos."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from experiments.jepa_cosmos.adapter import build_adapter
from experiments.jepa_cosmos.common import (
    atomic_json_dump,
    cosmos_normalize,
    imagenet_normalize,
    load_config,
    project_path,
    require_cuda,
    require_secret,
    resize_center_crop_uint8,
)
from experiments.jepa_cosmos.losses import latent_alignment_loss
from experiments.jepa_cosmos.media import write_comparison_video, write_video
from experiments.jepa_cosmos.models import (
    CosmosContinuousTokenizer,
    FactorJEPAWorldModel,
    validate_model_geometry,
)
from experiments.jepa_cosmos.tracking import log_file_artifact, log_video, start_wandb

import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.video_io import decode_video_bytes


def resolve_adapter_checkpoint(config: dict, value: str | None) -> Path:
    if value is not None:
        path = project_path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    local = project_path(config["experiment"]["output_dir"]) / "adapter_best.pt"
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    token = require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",))
    return Path(
        hf_hub_download(
            repo_id=config["huggingface"]["repo_id"],
            repo_type="model",
            filename="checkpoints/best.pt",
            token=token,
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
    data_cfg = config["data"]
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
    world = FactorJEPAWorldModel(config, device)
    cosmos = CosmosContinuousTokenizer(config, device, load_encoder=True, load_decoder=True)

    with tempfile.TemporaryDirectory(prefix="jepa_cosmos_infer_") as temporary_dir:
        raw = decode_video_bytes(
            input_path.read_bytes(),
            temporary_dir,
            input_path.name,
            data_cfg["num_frames"],
            cache_frames=False,
        )
    if raw is None:
        raise RuntimeError(f"Could not decode {input_path}")
    cropped = resize_center_crop_uint8(raw, data_cfg["crop_size"])
    normalized = imagenet_normalize(cropped).unsqueeze(0)
    context = normalized[:, :data_cfg["context_frames"]]
    anchor_rgb = cropped[data_cfg["context_frames"] - 1:data_cfg["context_frames"]]
    future_rgb = cropped[data_cfg["context_frames"]:]
    future_frames = future_rgb.shape[0]
    anchored_rgb = cropped[data_cfg["context_frames"] - 1:]
    anchor_video = cosmos_normalize(anchor_rgb).unsqueeze(0).to(device)
    anchored_video = cosmos_normalize(anchored_rgb).unsqueeze(0).to(device)

    with torch.no_grad(), torch.autocast(
        "cuda", dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[
            config["training"]["mixed_precision"]
        ]
    ):
        jepa_target = world.target_future(normalized)
        jepa_prediction = world.predict_future(context)
        cosmos_anchor = cosmos.encode(anchor_video)
        anchored_target = cosmos.encode(anchored_video)
        cosmos_target = anchored_target[:, :, 1:]
        target_shape = tuple(cosmos_target.shape[-3:])
        oracle_latent = adapter(jepa_target, target_shape)
        predicted_latent = adapter(jepa_prediction, target_shape)
    _, oracle_parts = latent_alignment_loss(
        oracle_latent,
        cosmos_target,
        config["loss"]["latent_l1"],
        config["loss"]["latent_cosine"],
    )
    _, predicted_parts = latent_alignment_loss(
        predicted_latent,
        cosmos_target,
        config["loss"]["latent_l1"],
        config["loss"]["latent_cosine"],
    )
    cosmos_reconstruction = cosmos.decode_anchored(
        cosmos_anchor, cosmos_target, future_frames
    )[0]
    oracle_video = cosmos.decode_anchored(
        cosmos_anchor, oracle_latent, future_frames
    )[0]
    predicted_video = cosmos.decode_anchored(
        cosmos_anchor, predicted_latent, future_frames
    )[0]

    context_path = output_dir / "context.mp4"
    ground_truth_path = output_dir / "ground_truth_future.mp4"
    predicted_path = output_dir / "predicted_future.mp4"
    comparison_path = output_dir / "comparison.mp4"
    write_video(context_path, cropped[:data_cfg["context_frames"]].permute(1, 0, 2, 3), args.fps)
    write_video(ground_truth_path, future_rgb.permute(1, 0, 2, 3), args.fps)
    write_video(predicted_path, ((predicted_video + 1) * 127.5).round().clamp(0, 255).to(torch.uint8), args.fps)
    write_comparison_video(
        comparison_path,
        future_rgb.permute(1, 0, 2, 3),
        cosmos_reconstruction,
        oracle_video,
        predicted_video,
        args.fps,
    )
    metrics = {
        "oracle_latent_l1": float(oracle_parts["latent_l1"]),
        "oracle_latent_cosine": float(oracle_parts["latent_cosine"]),
        "predicted_latent_l1": float(predicted_parts["latent_l1"]),
        "predicted_latent_cosine": float(predicted_parts["latent_cosine"]),
        "input_video": str(input_path),
        "adapter_checkpoint": str(adapter_path),
        "cosmos_anchor": "last_observed_context_frame",
    }
    metrics_path = output_dir / "metrics.json"
    atomic_json_dump(metrics, metrics_path)
    if not args.no_wandb:
        run = start_wandb(config, "inference", run_name=f"infer-{input_path.stem}")
        run.log({f"inference/{key}": value for key, value in metrics.items() if isinstance(value, float)})
        log_video(run, "inference/context", context_path)
        log_video(run, "inference/ground_truth_future", ground_truth_path)
        log_video(run, "inference/predicted_future", predicted_path)
        log_video(run, "inference/comparison", comparison_path)
        log_file_artifact(run, f"inference-{input_path.stem}", "inference", [metrics_path])
        run.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
