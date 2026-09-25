"""Labeled one-frame comparisons with the observed context video."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


HEADERS = (
    "Ground truth frame 15",
    "Cosmos-CI reconstruction",
    "Oracle: true JEPA",
    "Causal: predicted JEPA",
)

VIDEO_HEADERS = (
    "Observed context: frames 1-14",
    *HEADERS,
)


def _to_image(value: torch.Tensor) -> Image.Image:
    tensor = value.detach().cpu()
    if tensor.dtype != torch.uint8:
        tensor = ((tensor.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode="RGB")


def save_comparison(
    path: Path,
    key: str,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    oracle: torch.Tensor,
    prediction: torch.Tensor,
) -> None:
    target_image = target.detach().cpu()
    if target_image.dtype == torch.uint8:
        target_image = target_image.float() / 127.5 - 1.0
    images = [_to_image(item) for item in (target_image, reconstruction, oracle, prediction)]
    panel_width, panel_height = images[0].size
    header_height = 36
    footer_height = 28
    canvas = Image.new(
        "RGB", (panel_width * len(images), panel_height + header_height + footer_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for index, (header, image) in enumerate(zip(HEADERS, images)):
        left = index * panel_width
        canvas.paste(image, (left, header_height))
        box = draw.textbbox((0, 0), header, font=font)
        draw.text(
            (left + max(4, (panel_width - (box[2] - box[0])) // 2), 10),
            header,
            fill="black",
            font=font,
        )
        if index:
            draw.line((left, 0, left, canvas.height), fill=(80, 80, 80), width=2)
    draw.text(
        (8, header_height + panel_height + 8),
        f"Context frames 1-14 | target frame 15 | predicted tubelet frames 15-16 | {key}",
        fill="black",
        font=ImageFont.load_default(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_context_comparison_video(
    path: Path,
    context: torch.Tensor,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    oracle: torch.Tensor,
    prediction: torch.Tensor,
    fps: int = 4,
) -> None:
    """Write a five-panel video with moving context and four static readouts.

    The context tensor is (T, 3, H, W). The four frame-15 panels are repeated
    for every context timestep so all of the evidence for a prediction is
    visible together in a single W&B video.
    """
    if context.ndim != 4 or context.shape[1] != 3:
        raise ValueError(f"Expected context shape (T,3,H,W), got {tuple(context.shape)}")
    if context.shape[0] == 0:
        raise ValueError("Context video contains no frames")
    if fps <= 0:
        raise ValueError("fps must be positive")

    target_image = target.detach().cpu()
    if target_image.dtype == torch.uint8:
        target_image = target_image.float() / 127.5 - 1.0
    static_images = [
        _to_image(item) for item in (target_image, reconstruction, oracle, prediction)
    ]
    panel_width, panel_height = static_images[0].size
    if any(image.size != (panel_width, panel_height) for image in static_images[1:]):
        raise ValueError("All static comparison panels must have identical dimensions")

    header_height = 32
    frames: list[np.ndarray] = []
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for context_frame in context:
        context_image = _to_image(context_frame)
        if context_image.size != (panel_width, panel_height):
            raise ValueError("Context and frame-15 panels must have identical dimensions")
        images = [context_image, *static_images]
        canvas = Image.new(
            "RGB",
            (panel_width * len(images), panel_height + header_height),
            (18, 18, 18),
        )
        draw = ImageDraw.Draw(canvas)
        for index, (header, image) in enumerate(zip(VIDEO_HEADERS, images)):
            left = index * panel_width
            canvas.paste(image, (left, header_height))
            box = draw.textbbox((0, 0), header, font=font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]
            draw.text(
                (
                    left + max(4, (panel_width - text_width) // 2),
                    max(2, (header_height - text_height) // 2 - box[1]),
                ),
                header,
                fill="white",
                font=font,
            )
            if index:
                draw.line((left, 0, left, canvas.height), fill=(235, 235, 235), width=2)
        frames.append(np.asarray(canvas))

    path.parent.mkdir(parents=True, exist_ok=True)
    # 5*384 by (384+32) is 1920x416, already divisible by H.264's 16px
    # macroblock and therefore does not trigger imageio's implicit resizing.
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)


def save_rgb(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = image.detach().cpu()
    if tensor.dtype != torch.uint8:
        tensor = ((tensor.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode="RGB").save(path)
