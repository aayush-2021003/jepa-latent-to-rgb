"""Labeled two-frame comparison images."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


HEADERS = (
    "Last context",
    "GT frame 15",
    "Oracle frame 15",
    "Predicted frame 15",
    "GT frame 16",
    "Oracle frame 16",
    "Predicted frame 16",
)


def _to_image(value: torch.Tensor) -> Image.Image:
    tensor = value.detach().cpu()
    if tensor.dtype != torch.uint8:
        tensor = (tensor.float().clamp(0, 1) * 255.0).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def save_comparison(
    path: Path,
    key: str,
    last_context: torch.Tensor,
    target: torch.Tensor,
    oracle: torch.Tensor,
    prediction: torch.Tensor,
) -> None:
    images = [_to_image(item) for item in (
        last_context, target[0], oracle[0], prediction[0], target[1], oracle[1], prediction[1]
    )]
    panel_width, panel_height = images[0].size
    header_height = 32
    footer_height = 28
    canvas = Image.new(
        "RGB",
        (panel_width * len(images), panel_height + header_height + footer_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (header, image) in enumerate(zip(HEADERS, images)):
        left = index * panel_width
        canvas.paste(image, (left, header_height))
        box = draw.textbbox((0, 0), header, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (left + max(4, (panel_width - text_width) // 2), 10),
            header,
            fill="black",
            font=font,
        )
    footer = f"Held-out frames 15-16 | context frames 1-14 | {key}"
    draw.text((8, header_height + panel_height + 8), footer, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
