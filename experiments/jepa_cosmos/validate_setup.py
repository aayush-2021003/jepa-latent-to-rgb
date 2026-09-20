"""Fail-loud preflight checks for data, dependencies, credentials, and geometry."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import tarfile

import torch

from experiments.jepa_cosmos.adapter import build_adapter
from experiments.jepa_cosmos.common import load_config, project_path, require_secret
from experiments.jepa_cosmos.models import validate_model_geometry


def module_status(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def validate_local_dataset(config: dict) -> dict:
    root = project_path(config["data"]["local_root"])
    result = {}
    sources = {}
    for split in ("train", "val"):
        split_root = root / split
        manifest_path = split_root / "manifest.json"
        subset_path = split_root / f"{split}.json"
        if not manifest_path.is_file() or not subset_path.is_file():
            result[split] = "not prepared"
            continue
        manifest = json.loads(manifest_path.read_text())
        subset = json.loads(subset_path.read_text())
        shards = sorted((split_root / "m00d_download_subset").glob("subset-*.tar"))
        if len(subset["clip_keys"]) != manifest["n"]:
            raise RuntimeError(f"{split}: subset/manifest count mismatch")
        if not shards:
            raise RuntimeError(f"{split}: no local TAR shards")
        with tarfile.open(shards[0], "r") as archive:
            if not any(member.name.endswith(".mp4") for member in archive.getmembers()):
                raise RuntimeError(f"{split}: first TAR has no MP4 members")
        sources[split] = {key.split("/")[-2][:11] for key in subset["clip_keys"]}
        result[split] = {"samples": manifest["n"], "shards": len(shards)}
    if "train" in sources and "val" in sources:
        overlap = sources["train"] & sources["val"]
        if overlap:
            raise RuntimeError(f"Train/validation source overlap: {sorted(overlap)[:3]}")
        result["source_overlap"] = 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_model_geometry(config)
    dependencies = {
        name: module_status(name)
        for name in ("torch", "yaml", "wandb", "huggingface_hub", "cosmos_tokenizer", "av")
    }
    missing = [name for name, available in dependencies.items() if not available]
    if missing:
        raise RuntimeError(f"Missing dependencies: {missing}")

    factor_path = project_path(config["factorjepa"]["checkpoint_path"])
    cosmos_dir = project_path(config["cosmos"]["checkpoint_dir"])
    assets = {
        "factorjepa_checkpoint": factor_path.is_file(),
        "cosmos_encoder": (cosmos_dir / config["cosmos"]["encoder_file"]).is_file(),
        "cosmos_decoder": (cosmos_dir / config["cosmos"]["decoder_file"]).is_file(),
        "vjepa2_source": (project_path("deps/vjepa2") / "src").is_dir(),
    }

    adapter = build_adapter(config)
    dummy = torch.randn(
        1,
        config["adapter"]["input_dim"],
        4,
        24,
        24,
    )
    future_frames = config["data"]["num_frames"] - config["data"]["context_frames"]
    future_slots = future_frames // config["cosmos"]["temporal_compression"]
    cosmos_spatial = config["data"]["crop_size"] // 8
    target_geometry = (future_slots, cosmos_spatial, cosmos_spatial)
    with torch.no_grad():
        output = adapter(dummy, target_geometry)
    expected_output = (1, config["cosmos"]["latent_channels"], *target_geometry)
    if output.shape != expected_output:
        raise RuntimeError(f"Adapter geometry smoke failed: {tuple(output.shape)}")

    online = "not requested"
    if args.online:
        hf_token = require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",))
        wandb_key = require_secret("WANDB_API_KEY")
        from huggingface_hub import HfApi
        import wandb

        hf_api = HfApi(token=hf_token)
        viewer = hf_api.whoami()
        expected_owner = config["huggingface"]["repo_id"].split("/", 1)[0]
        if viewer["name"].lower() != expected_owner.lower():
            raise RuntimeError(
                f"HF_TOKEN belongs to {viewer['name']}, but repo_id owner is {expected_owner}"
            )
        hf_api.list_repo_files(config["data"]["archive_repo"], repo_type="dataset")
        wandb.login(key=wandb_key, relogin=True)
        wandb_api = wandb.Api()
        next(
            iter(wandb_api.projects(entity=config["tracking"]["wandb_entity"], per_page=1)),
            None,
        )
        online = {
            "huggingface_user": viewer["name"],
            "denseworld_archive_access": True,
            "wandb_entity_access": True,
        }

    report = {
        "dependencies": dependencies,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "assets": assets,
        "dataset": validate_local_dataset(config),
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "adapter_output_shape": list(output.shape),
        "online": online,
        "secrets_embedded_in_config": bool(
            re.search(r"wandb_v1_[A-Za-z0-9_]+|hf_[A-Za-z0-9]{20,}", json.dumps(config))
        ),
    }
    print(json.dumps(report, indent=2))
    if report["secrets_embedded_in_config"]:
        raise RuntimeError("A secret-like value was embedded in the YAML config")


if __name__ == "__main__":
    main()
