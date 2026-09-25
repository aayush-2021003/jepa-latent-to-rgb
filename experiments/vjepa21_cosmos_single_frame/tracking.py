"""W&B tracking and Hugging Face publication for the single-frame adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.jepa_cosmos.tracking import log_file_artifact, start_wandb
from experiments.vjepa21_cosmos_single_frame.common import require_secret


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
        commit_message=f"Update {alias} V-JEPA 2.1-to-Cosmos-CI adapter",
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
        "pipeline_tag: video-to-image\n"
        "tags:\n- v-jepa-2.1\n- cosmos-tokenizer\n- world-models\n---\n\n"
        "# V-JEPA 2.1 predicted tubelet to Cosmos-CI frame adapter\n\n"
        "Frames 1-14 are observed. Frozen V-JEPA 2.1 predicts one joint tubelet "
        "covering frames 15-16. This adapter reads that predicted tubelet and maps "
        "it to the frozen Cosmos-CI8x8 latent of frame 15 only. Training uses latent "
        "L1 plus 0.1 cosine distance. This is a single-frame readout, not a change "
        "to V-JEPA's native two-frame tubelet size. Upstream weights are excluded.\n\n"
        f"- V-JEPA checkpoint: `{config['vjepa21']['checkpoint_url']}`\n"
        f"- Cosmos image tokenizer: `{config['cosmos']['model_id']}`\n"
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


__all__ = ["log_file_artifact", "push_checkpoint_to_hub", "start_wandb"]
