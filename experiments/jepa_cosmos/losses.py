"""Losses for latent alignment and optional frozen-decoder supervision."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_alignment_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float,
    cosine_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape:
        raise ValueError(f"Latent shape mismatch: {prediction.shape} vs {target.shape}")
    l1 = F.l1_loss(prediction.float(), target.float())
    cosine = 1.0 - F.cosine_similarity(
        prediction.float().flatten(1), target.float().flatten(1), dim=1
    ).mean()
    total = l1_weight * l1 + cosine_weight * cosine
    return total, {"latent_l1": l1, "latent_cosine": cosine}


def decoded_video_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    rgb_l1_weight: float,
    temporal_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Both inputs are ``(B,3,T,H,W)`` in ``[-1,1]``."""
    rgb_l1 = F.l1_loss(prediction.float(), target.float())
    pred_delta = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_delta = target[:, :, 1:] - target[:, :, :-1]
    temporal = F.l1_loss(pred_delta.float(), target_delta.float())
    total = rgb_l1_weight * rgb_l1 + temporal_weight * temporal
    return total, {"rgb_l1": rgb_l1, "temporal_l1": temporal}


class PerceptualVideoLoss(torch.nn.Module):
    """LPIPS over video frames, instantiated only when its weight is non-zero."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "loss.perceptual is non-zero but lpips is not installed. "
                "Re-run the GPU environment setup."
            ) from error
        self.model = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = prediction.shape
        pred_frames = prediction.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        target_frames = target.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        return self.model(pred_frames.float(), target_frames.float()).mean()
