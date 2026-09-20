"""Strict W&B tracking and Hugging Face checkpoint publication."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.jepa_cosmos.common import require_secret


def start_wandb(config: dict, job_type: str, run_name: str | None = None):
    import wandb

    api_key = require_secret("WANDB_API_KEY")
    wandb.login(key=api_key, relogin=True)
    tracking = config["tracking"]
    return wandb.init(
        entity=tracking["wandb_entity"],
        project=tracking["wandb_project"],
        name=run_name,
        job_type=job_type,
        config={key: value for key, value in config.items() if not key.startswith("_")},
        reinit=True,
    )


def log_file_artifact(run, name: str, artifact_type: str, paths: list[Path]) -> None:
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type)
    for path in paths:
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)


def log_video(run, key: str, path: Path) -> None:
    import wandb

    if not path.is_file():
        raise FileNotFoundError(path)
    # A file-backed MP4 already contains its frame rate; W&B ignores `fps` here.
    run.log({key: wandb.Video(str(path), format="mp4")})


def push_checkpoint_to_hub(
    config: dict,
    checkpoint: Path,
    metrics: dict[str, Any],
    alias: str,
) -> str:
    """Upload only the adapter checkpoint and small metadata to the user's model repo."""
    from huggingface_hub import HfApi

    token = require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",))
    hub_cfg = config["huggingface"]
    repo_id = hub_cfg["repo_id"]
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=hub_cfg["private"],
        exist_ok=True,
    )
    remote_name = f"checkpoints/{alias}.pt"
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=remote_name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {alias} FactorJEPA-to-Cosmos adapter",
    )
    metrics_path = checkpoint.parent / f"{alias}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    api.upload_file(
        path_or_fileobj=str(metrics_path),
        path_in_repo=f"metrics/{alias}.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {alias} validation metrics",
    )
    card_path = checkpoint.parent / "README_huggingface.md"
    card_path.write_text(
        "---\n"
        "library_name: pytorch\n"
        "pipeline_tag: video-to-video\n"
        "tags:\n"
        "- factorjepa\n"
        "- cosmos-tokenizer\n"
        "- world-models\n"
        "---\n\n"
        "# FactorJEPA to Cosmos future-latent adapter\n\n"
        "This repository contains the small trainable adapter that maps frozen "
        "FactorJEPA future tokens into NVIDIA Cosmos continuous video latents. "
        "It predicts two future latent slots; a separately encoded last-context-frame "
        "anchor is prepended only for causal 4k+1 decoding. "
        "It does not contain the upstream FactorJEPA or Cosmos weights.\n\n"
        f"- Base FactorJEPA checkpoint: `{config['factorjepa']['checkpoint_repo']}/"
        f"{config['factorjepa']['checkpoint_file']}`\n"
        f"- Cosmos tokenizer: `{config['cosmos']['model_id']}`\n"
        f"- Best/latest metric payload: `metrics/{alias}.json`\n"
        "- Training code: `experiments/jepa_cosmos` in the FactorJEPA repository.\n"
    )
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update adapter model card",
    )
    config_path = Path(config["_config_path"])
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.yaml",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update reproducible experiment config",
    )
    return f"https://huggingface.co/{repo_id}/blob/main/{remote_name}"
