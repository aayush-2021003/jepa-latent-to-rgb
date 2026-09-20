"""Small video writers used for validation and inference deliverables."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


COMPARISON_LABELS = (
    "Ground Truth Future",
    "Cosmos Reconstruction",
    "Oracle: True JEPA",
    "Causal: Predicted JEPA",
)


def _label_comparison_frame(frame: np.ndarray, panel_width: int) -> np.ndarray:
    """Add a persistent header bar above each comparison column."""
    header_height = 36
    height, width, channels = frame.shape
    canvas = Image.new("RGB", (width, height + header_height), color=(18, 18, 18))
    canvas.paste(Image.fromarray(frame), (0, header_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for index, label in enumerate(COMPARISON_LABELS):
        left = index * panel_width
        right = left + panel_width
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        x = left + max(8, (right - left - text_width) // 2)
        y = max(2, (header_height - text_height) // 2 - box[1])
        draw.text((x, y), label, fill=(255, 255, 255), font=font)
        if index:
            draw.line((left, 0, left, height + header_height), fill=(235, 235, 235), width=2)
    result = np.asarray(canvas)
    if channels != 3:
        raise ValueError(f"Expected RGB comparison frame, got {channels} channels")
    return result


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
    panel_width = arrays[0].shape[2]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("All comparison videos must have identical dimensions")
    frames = np.concatenate(arrays, axis=2)
    frames = np.stack(
        [_label_comparison_frame(frame, panel_width) for frame in frames], axis=0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)
