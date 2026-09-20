"""Shared helpers for the isolated V-JEPA 2.1 two-frame experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.jepa_cosmos.common import (
    PROJECT_ROOT,
    atomic_json_dump,
    atomic_torch_save,
    checkpoint_fingerprint,
    imagenet_normalize,
    load_dotenv_if_available,
    project_path,
    require_cuda,
    require_secret,
    resize_center_crop_uint8,
    seed_everything,
    volume_from_tokens,
)


REQUIRED_SECTIONS = {
    "experiment",
    "data",
    "vjepa21",
    "jepa_wms_decoder",
    "adapter",
    "training",
    "loss",
    "tracking",
    "huggingface",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load this experiment's schema without depending on the Cosmos schema."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise KeyError(f"Experiment config is missing sections: {sorted(missing)}")
    config["_config_path"] = str(config_path)
    return config
