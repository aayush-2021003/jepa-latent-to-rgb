"""Fail-fast validation for the V-JEPA 2.1 two-frame experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.vjepa21_jepawms.adapter import VJEPA21ToJEPAWMSAdapter
from experiments.vjepa21_jepawms.common import load_config, project_path
from experiments.vjepa21_jepawms.data import CACHE_SCHEMA
from experiments.vjepa21_jepawms.models import validate_geometry


def validate_cache(config: dict, required: bool) -> dict:
    result = {}
    for split, expected in (
        ("train", config["data"]["train_samples"]),
        ("val", config["data"]["val_samples"]),
    ):
        manifest_path = project_path(config["data"]["cache_root"]) / split / "manifest.json"
        if not manifest_path.is_file():
            if required:
                raise FileNotFoundError(manifest_path)
            result[split] = "not built"
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("metadata", {}).get("schema") != CACHE_SCHEMA:
            raise RuntimeError(f"Wrong cache schema: {manifest_path}")
        if manifest["samples"] != expected:
            raise RuntimeError(
                f"{split} cache has {manifest['samples']} samples, expected {expected}"
            )
        missing = [
            item["file"]
            for item in manifest["shards"]
            if not (manifest_path.parent / item["file"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {split} cache shards: {missing[:3]}")
        result[split] = manifest["samples"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_geometry(config)

    adapter = VJEPA21ToJEPAWMSAdapter(
        levels=config["adapter"]["levels"],
        level_dim=config["adapter"]["level_dim"],
        output_dim=config["adapter"]["output_dim"],
        residual_blocks=config["adapter"]["residual_blocks"],
        expansion=config["adapter"]["expansion"],
        dropout=config["adapter"]["dropout"],
        output_frames=config["adapter"]["output_frames"],
    )
    parameter_count = sum(parameter.numel() for parameter in adapter.parameters())

    assets = {
        "vjepa21_checkpoint": project_path(config["vjepa21"]["checkpoint_path"]),
        "jepa_wms_decoder": project_path(
            config["jepa_wms_decoder"]["checkpoint_path"]
        ),
        "vjepa2_dependency": project_path(config["vjepa21"]["dependency_root"]),
        "jepa_wms_dependency": project_path(
            config["jepa_wms_decoder"]["dependency_root"]
        ),
    }
    asset_status = {}
    for name, path in assets.items():
        exists = path.exists()
        if args.require_assets and not exists:
            raise FileNotFoundError(f"Missing required asset {name}: {path}")
        asset_status[name] = str(path) if exists else "missing"

    report = {
        "status": "ok",
        "task": "frames 1-14 -> V-JEPA 2.1 predicts tubelet 15-16 -> RGB frames 15-16",
        "adapter_parameters": parameter_count,
        "assets": asset_status,
        "cache": validate_cache(config, args.require_cache),
        "cuda_available": torch.cuda.is_available(),
        "output_dir": str(project_path(config["experiment"]["output_dir"])),
        "wandb_run_name": config["tracking"]["run_name"],
        "huggingface_repo": config["huggingface"]["repo_id"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
