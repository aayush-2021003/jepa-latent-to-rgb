"""Shared, dependency-light helpers for the JEPA-to-Cosmos experiment."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the experiment YAML and resolve project-relative paths."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
    required = {
        "experiment", "data", "factorjepa", "cosmos", "adapter",
        "training", "loss", "tracking", "huggingface",
    }
    missing = required - set(cfg)
    if missing:
        raise KeyError(f"Experiment config is missing sections: {sorted(missing)}")
    cfg["_config_path"] = str(config_path)
    return cfg


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def require_secret(name: str, aliases: tuple[str, ...] = ()) -> str:
    """Read a secret from the environment without ever displaying its value."""
    load_dotenv_if_available()
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value:
            return value
    names = ", ".join((name, *aliases))
    raise RuntimeError(f"Missing required environment variable ({names}).")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This stage requires an NVIDIA GPU; CUDA is not available.")
    return torch.device("cuda")


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp{os.getpid()}{path.suffix}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp{os.getpid()}{path.suffix}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_fingerprint(path: str | Path) -> str:
    """Cheap identity stamp that avoids hashing a multi-GB checkpoint."""
    path = Path(path)
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def clip_key_from_metadata(metadata: dict[str, Any]) -> str:
    return "/".join(
        (metadata["section"], metadata["video_id"], metadata["source_file"])
    )


def resize_center_crop_uint8(frames: torch.Tensor, crop: int) -> torch.Tensor:
    """Convert ``(T,C,H,W)`` uint8 frames to a shared square RGB view."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"Expected (T,3,H,W), got {tuple(frames.shape)}")
    frames_float = frames.float()
    height, width = frames.shape[-2:]
    scale = crop / min(height, width)
    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    resized = F.interpolate(
        frames_float,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (new_height - crop) // 2
    left = (new_width - crop) // 2
    return resized[:, :, top:top + crop, left:left + crop].round().clamp(0, 255).to(torch.uint8)


def imagenet_normalize(frames: torch.Tensor) -> torch.Tensor:
    """``uint8 (T,3,H,W)`` to FactorJEPA's ImageNet-normalized float tensor."""
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (frames.float() / 255.0 - mean) / std


def cosmos_normalize(frames: torch.Tensor) -> torch.Tensor:
    """``uint8 (T,3,H,W)`` to ``(3,T,H,W)`` in Cosmos' ``[-1,1]`` range."""
    return (frames.float() / 127.5 - 1.0).permute(1, 0, 2, 3).contiguous()


def volume_from_tokens(
    tokens: torch.Tensor,
    temporal_slots: int,
    spatial_height: int,
    spatial_width: int,
) -> torch.Tensor:
    """Convert ``(B,N,D)`` JEPA token order to ``(B,D,T,H,W)``."""
    batch, token_count, dim = tokens.shape
    expected = temporal_slots * spatial_height * spatial_width
    if token_count != expected:
        raise ValueError(f"Token count {token_count} does not match grid size {expected}")
    return (
        tokens.reshape(batch, temporal_slots, spatial_height, spatial_width, dim)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )


def tokens_from_volume(volume: torch.Tensor) -> torch.Tensor:
    """Convert ``(B,D,T,H,W)`` to ``(B,N,D)`` in JEPA token order."""
    return volume.permute(0, 2, 3, 4, 1).flatten(1, 3).contiguous()

