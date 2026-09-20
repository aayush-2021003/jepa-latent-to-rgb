"""Small video writers used for validation and inference deliverables."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def cosmos_to_uint8(video: torch.Tensor) -> torch.Tensor:
    """``(3,T,H,W)`` or ``(B,3,T,H,W)`` in ``[-1,1]`` to uint8."""
    return ((video.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


def write_video(path: Path, video: torch.Tensor, fps: int) -> None:
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("write_video accepts a single video")
        video = video[0]
    frames = video.permute(1, 2, 3, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)


def write_comparison_video(
    path: Path,
    ground_truth: torch.Tensor,
    cosmos_reconstruction: torch.Tensor,
    oracle_adapter: torch.Tensor,
    predicted_future: torch.Tensor,
    fps: int,
) -> None:
    videos = [ground_truth, cosmos_reconstruction, oracle_adapter, predicted_future]
    arrays = []
    for video in videos:
        if video.ndim == 5:
            video = video[0]
        if video.dtype != torch.uint8:
            video = cosmos_to_uint8(video)
        arrays.append(video.permute(1, 2, 3, 0).cpu().numpy())
    frames = np.concatenate(arrays, axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)

