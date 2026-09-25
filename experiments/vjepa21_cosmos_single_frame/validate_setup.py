"""Fail-fast checks for the V-JEPA 2.1/Cosmos-CI single-frame experiment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re

import torch

from experiments.jepa_cosmos.validate_setup import validate_local_dataset
from experiments.vjepa21_cosmos_single_frame.adapter import build_adapter
from experiments.vjepa21_cosmos_single_frame.common import load_config, project_path, require_secret
from experiments.vjepa21_cosmos_single_frame.data import CACHE_SCHEMA
from experiments.vjepa21_cosmos_single_frame.models import validate_model_geometry


def cache_status(config: dict, required: bool) -> dict:
    result = {}
    for split, expected in (
        ("train", int(config["data"]["train_samples"])),
        ("val", int(config["data"]["val_samples"])),
    ):
        manifest_path = project_path(config["data"]["cache_root"]) / split / "manifest.json"
        if not manifest_path.is_file():
            if required:
                raise FileNotFoundError(manifest_path)
            result[split] = "not built"
            continue
        manifest = json.loads(manifest_path.read_text())
        metadata = manifest.get("metadata", {})
        if metadata.get("schema") != CACHE_SCHEMA:
            raise RuntimeError(f"Wrong cache schema: {manifest_path}")
        if int(manifest["samples"]) != expected:
            raise RuntimeError(
                f"{split} cache has {manifest['samples']} samples, expected {expected}"
            )
        if metadata.get("contains_jepa_predicted") is not True:
            raise RuntimeError(f"{split} cache lacks predicted V-JEPA features")
        if metadata.get("target_frame_number") != 15:
            raise RuntimeError(f"{split} cache does not target frame 15")
        if split == "val" and metadata.get("contains_jepa_target") is not True:
            raise RuntimeError("Validation cache lacks oracle V-JEPA features")
        if split == "val" and metadata.get("contains_target_rgb") is not True:
            raise RuntimeError("Validation cache lacks frame-15 RGB targets")
        if (
            split == "val"
            and config["tracking"].get("log_context_panel", False)
            and metadata.get("preview_contains_context_rgb") is not True
        ):
            raise RuntimeError("Validation cache lacks 14-frame RGB context previews")
        missing = [
            row["file"]
            for row in manifest["shards"]
            if not (manifest_path.parent / row["file"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {split} cache shards: {missing[:3]}")
        if split == "val" and config["tracking"].get("log_context_panel", False):
            preview_path = manifest_path.parent / "preview.pt"
            if not preview_path.is_file():
                raise FileNotFoundError(preview_path)
            preview = torch.load(preview_path, map_location="cpu", weights_only=False)
            if not preview or any("context_rgb" not in sample for sample in preview):
                raise RuntimeError("Validation preview cache lacks 14-frame RGB context")
        result[split] = manifest["samples"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_model_geometry(config)
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "yaml", "wandb", "huggingface_hub", "cosmos_tokenizer", "av")
    }
    adapter = build_adapter(config)
    grid = int(config["data"]["crop_size"]) // int(config["vjepa21"]["patch_size"])
    latent_grid = int(config["data"]["crop_size"]) // int(config["cosmos"]["spatial_compression"])
    dummy = torch.randn(1, int(config["adapter"]["input_dim"]), 1, grid, grid)
    with torch.no_grad():
        output = adapter(dummy, (latent_grid, latent_grid))
    expected = (1, int(config["cosmos"]["latent_channels"]), latent_grid, latent_grid)
    if tuple(output.shape) != expected:
        raise RuntimeError(f"Adapter geometry failed: {tuple(output.shape)} != {expected}")

    vjepa_path = project_path(config["vjepa21"]["checkpoint_path"])
    cosmos_dir = project_path(config["cosmos"]["checkpoint_dir"])
    assets = {
        "vjepa21_checkpoint": vjepa_path.is_file(),
        "vjepa21_source": (
            project_path(config["vjepa21"]["dependency_root"])
            / "app"
            / "vjepa_2_1"
            / "models"
        ).is_dir(),
        "cosmos_ci_encoder": (cosmos_dir / config["cosmos"]["encoder_file"]).is_file(),
        "cosmos_ci_decoder": (cosmos_dir / config["cosmos"]["decoder_file"]).is_file(),
    }
    if args.require_assets and not all(assets.values()):
        raise RuntimeError(f"Missing required assets: {assets}")
    online = "not requested"
    if args.online:
        from huggingface_hub import HfApi
        import wandb

        hf_token = require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",))
        wandb_key = require_secret("WANDB_API_KEY")
        viewer = HfApi(token=hf_token).whoami()
        expected_owner = config["huggingface"]["repo_id"].split("/", 1)[0]
        if viewer["name"].lower() != expected_owner.lower():
            raise RuntimeError(
                f"HF token belongs to {viewer['name']}, expected {expected_owner}"
            )
        wandb.login(key=wandb_key, relogin=True)
        online = {"huggingface_user": viewer["name"], "wandb_login": True}
    report = {
        "status": "ok",
        "task": "frames 1-14 -> predicted tubelet 15-16 -> Cosmos-CI latent of frame 15",
        "warning": "single-frame readout from a native two-frame V-JEPA prediction",
        "training_loss": "frame15 latent_l1 + 0.1 * frame15 latent_cosine",
        "dependencies": dependencies,
        "assets": assets,
        "dataset": validate_local_dataset(config),
        "cache": cache_status(config, args.require_cache),
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "adapter_input_shape": list(dummy.shape),
        "adapter_output_shape": list(output.shape),
        "cuda_available": torch.cuda.is_available(),
        "online": online,
        "secrets_embedded_in_config": bool(
            re.search(r"wandb_v1_[A-Za-z0-9_]+|hf_[A-Za-z0-9]{20,}", json.dumps(config))
        ),
    }
    print(json.dumps(report, indent=2))
    if report["secrets_embedded_in_config"]:
        raise RuntimeError("A secret-like value was embedded in the config")


if __name__ == "__main__":
    main()
