"""Labeled one-frame comparison panels."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


HEADERS = (
    "Ground truth frame 15",
    "Cosmos-CI reconstruction",
    "Oracle: true JEPA",
    "Causal: predicted JEPA",
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


def save_rgb(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = image.detach().cpu()
    if tensor.dtype != torch.uint8:
        tensor = ((tensor.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode="RGB").save(path)
