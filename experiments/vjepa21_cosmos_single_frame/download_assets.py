"""Download the official V-JEPA 2.1 and Cosmos-CI checkpoints."""
from __future__ import annotations

import argparse
import json

from experiments.jepa_cosmos.tracking import log_file_artifact, start_wandb
from experiments.vjepa21_jepawms.download_assets import download_url
from experiments.vjepa21_cosmos_single_frame.common import (
    atomic_json_dump,
    load_config,
    project_path,
    require_secret,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    vjepa = config["vjepa21"]
    dependency = project_path(vjepa["dependency_root"])
    if not (dependency / "app" / "vjepa_2_1" / "models").is_dir():
        raise RuntimeError(
            f"V-JEPA 2 dependency is missing at {dependency}. "
            "Run bash setup_env_uv.sh --gpu --from-wheels first."
        )
    vjepa_path = project_path(vjepa["checkpoint_path"])
    download_url(vjepa["checkpoint_url"], vjepa_path)

    from huggingface_hub import snapshot_download

    token = require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",))
    cosmos = config["cosmos"]
    cosmos_dir = project_path(cosmos["checkpoint_dir"])
    snapshot_download(
        repo_id=cosmos["model_id"],
        token=token,
        local_dir=str(cosmos_dir),
        allow_patterns=[cosmos["encoder_file"], cosmos["decoder_file"], "README.md"],
    )
    expected = [
        vjepa_path,
        cosmos_dir / cosmos["encoder_file"],
        cosmos_dir / cosmos["decoder_file"],
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Asset download completed with missing files: {missing}")
    summary = {
        "vjepa21": {"path": str(vjepa_path), "bytes": vjepa_path.stat().st_size},
        "cosmos_ci": {
            "model_id": cosmos["model_id"],
            "encoder": str(cosmos_dir / cosmos["encoder_file"]),
            "decoder": str(cosmos_dir / cosmos["decoder_file"]),
        },
    }
    summary_path = project_path(config["experiment"]["output_dir"]) / "asset_summary.json"
    atomic_json_dump(summary, summary_path)
    if not args.no_wandb:
        run = start_wandb(config, "asset-download", f"{config['tracking']['run_name']}-assets")
        log_file_artifact(run, "vjepa21-cosmos-ci-assets", "metadata", [summary_path])
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
