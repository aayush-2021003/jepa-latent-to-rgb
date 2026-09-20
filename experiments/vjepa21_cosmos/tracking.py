"""W&B tracking and Hugging Face publication for V-JEPA 2.1/Cosmos."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.vjepa21_cosmos.common import require_secret
from experiments.jepa_cosmos.tracking import log_file_artifact, log_video, start_wandb


def push_checkpoint_to_hub(
    config: dict, checkpoint: Path, metrics: dict[str, Any], alias: str
) -> str:
    from huggingface_hub import HfApi

    token = require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",))
    hub = config["huggingface"]
    api = HfApi(token=token)
    api.create_repo(
        repo_id=hub["repo_id"], repo_type="model", private=hub["private"], exist_ok=True
    )
    remote = f"checkpoints/{alias}.pt"
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=remote,
        repo_id=hub["repo_id"],
        repo_type="model",
        commit_message=f"Update {alias} V-JEPA 2.1-to-Cosmos adapter",
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
    card = checkpoint.parent / "README_huggingface.md"
    card.write_text(
        "---\n"
        "library_name: pytorch\n"
        "pipeline_tag: video-to-video\n"
        "tags:\n- v-jepa-2.1\n- cosmos-tokenizer\n- world-models\n---\n\n"
        "# V-JEPA 2.1 predicted-latent to Cosmos adapter\n\n"
        "This adapter maps two frozen V-JEPA 2.1 predicted tubelets covering "
        "four future frames into one Cosmos CV4 future latent slot. Frames 1-12 "
        "are context and frames 13-16 are predicted. Training uses latent L1 plus "
        "0.1 cosine distance; no RGB loss is used. Upstream weights are excluded.\n\n"
        f"- V-JEPA checkpoint: `{config['vjepa21']['checkpoint_url']}`\n"
        f"- Cosmos tokenizer: `{config['cosmos']['model_id']}`\n"
        f"- Metrics: `metrics/{alias}.json`\n"
    )
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=hub["repo_id"],
        repo_type="model",
        commit_message="Update model card",
    )
    api.upload_file(
        path_or_fileobj=config["_config_path"],
        path_in_repo="config.yaml",
        repo_id=hub["repo_id"],
        repo_type="model",
        commit_message="Update reproducible config",
    )
    return f"https://huggingface.co/{hub['repo_id']}/blob/main/{remote}"
