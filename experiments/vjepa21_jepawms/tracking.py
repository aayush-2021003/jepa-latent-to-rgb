"""Remote checkpoint publication for the V-JEPA 2.1 adapter."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.vjepa21_jepawms.common import require_secret


def push_checkpoint(config: dict, checkpoint: Path, metrics: dict, alias: str) -> str:
    from huggingface_hub import HfApi

    hub = config["huggingface"]
    token = require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",))
    api = HfApi(token=token)
    api.create_repo(
        repo_id=hub["repo_id"],
        repo_type="model",
        private=hub["private"],
        exist_ok=True,
    )
    remote_checkpoint = f"checkpoints/{alias}.pt"
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=remote_checkpoint,
        repo_id=hub["repo_id"],
        repo_type="model",
        commit_message=f"Update {alias} V-JEPA 2.1 two-frame adapter",
    )
    metrics_path = checkpoint.parent / f"{alias}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    api.upload_file(
        path_or_fileobj=str(metrics_path),
        path_in_repo=f"metrics/{alias}.json",
        repo_id=hub["repo_id"],
        repo_type="model",
        commit_message=f"Update {alias} validation metrics",
    )
    config_path = Path(config["_config_path"])
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.yaml",
        repo_id=hub["repo_id"],
        repo_type="model",
        commit_message="Update reproducible experiment config",
    )
    card = checkpoint.parent / "README_huggingface.md"
    card.write_text(
        "---\n"
        "library_name: pytorch\n"
        "pipeline_tag: image-to-image\n"
        "tags:\n- v-jepa-2.1\n- jepa-wms\n- world-model\n---\n\n"
        "# V-JEPA 2.1 predicted-latent two-frame adapter\n\n"
        "This repository contains only the learned adapter. It maps all four frozen "
        "V-JEPA 2.1 ViT-g predicted feature levels to the input space of Meta's frozen "
        "JEPA-WMs `vjepa2_vitg_256_INet` image decoder. Upstream model weights are not "
        "redistributed here. The task uses frames 1-14 as context, predicts the final "
        "two-frame V-JEPA tubelet, and supervises RGB frames 15 and 16.\n\n"
        f"Validation metrics: `metrics/{alias}.json`.\n"
    )
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=hub["repo_id"],
        repo_type="model",
        commit_message="Update model card",
    )
    return f"https://huggingface.co/{hub['repo_id']}/blob/main/{remote_checkpoint}"
