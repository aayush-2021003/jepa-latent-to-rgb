"""Download the exact FactorJEPA checkpoint and Cosmos encoder/decoder assets."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from experiments.jepa_cosmos.common import (
    atomic_json_dump,
    load_config,
    project_path,
    require_secret,
)
from experiments.jepa_cosmos.tracking import log_file_artifact, start_wandb


def materialize_cached_file(cached: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    # hf_hub_download commonly returns a relative symlink inside the HF cache.
    # Hard-linking that symlink itself into checkpoints/ makes its relative target
    # invalid, so always resolve it to the underlying blob first.
    cached = cached.resolve(strict=True)
    if destination.is_symlink():
        destination.unlink()
    temporary = destination.with_suffix(f".tmp{os.getpid()}{destination.suffix}")
    try:
        os.link(cached, temporary)
    except OSError:
        shutil.copy2(cached, temporary)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    token = require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",))

    from huggingface_hub import hf_hub_download, snapshot_download

    factor_cfg = config["factorjepa"]
    factor_path = project_path(factor_cfg["checkpoint_path"])
    cached_factor = Path(
        hf_hub_download(
            repo_id=factor_cfg["checkpoint_repo"],
            repo_type=factor_cfg["checkpoint_repo_type"],
            filename=factor_cfg["checkpoint_file"],
            token=token,
        )
    )
    materialize_cached_file(cached_factor, factor_path)

    cosmos_cfg = config["cosmos"]
    cosmos_dir = project_path(cosmos_cfg["checkpoint_dir"])
    snapshot_download(
        repo_id=cosmos_cfg["model_id"],
        token=token,
        local_dir=str(cosmos_dir),
        allow_patterns=[cosmos_cfg["encoder_file"], cosmos_cfg["decoder_file"], "README.md"],
    )
    expected = [
        factor_path,
        cosmos_dir / cosmos_cfg["encoder_file"],
        cosmos_dir / cosmos_cfg["decoder_file"],
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Asset download completed with missing files: {missing}")

    summary = {
        "factorjepa": {
            "repo": factor_cfg["checkpoint_repo"],
            "file": factor_cfg["checkpoint_file"],
            "local_path": str(factor_path),
            "bytes": factor_path.stat().st_size,
        },
        "cosmos": {
            "repo": cosmos_cfg["model_id"],
            "encoder": str(cosmos_dir / cosmos_cfg["encoder_file"]),
            "decoder": str(cosmos_dir / cosmos_cfg["decoder_file"]),
        },
    }
    summary_path = project_path(config["experiment"]["output_dir"]) / "asset_summary.json"
    atomic_json_dump(summary, summary_path)
    if not args.no_wandb:
        run = start_wandb(config, "asset-download")
        run.log({"assets/factorjepa_checkpoint_gb": factor_path.stat().st_size / 1e9})
        log_file_artifact(run, "jepa-cosmos-asset-manifest", "metadata", [summary_path])
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
