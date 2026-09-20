"""Training losses and image metrics for the two-frame adapter."""
from __future__ import annotations

import torch
from torch import nn


class PerceptualLoss(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "LPIPS is enabled but not installed. Run the GPU environment setup."
            ) from error
        self.model = lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model(prediction * 2.0 - 1.0, target * 2.0 - 1.0).mean()


def feature_statistics_loss(features: torch.Tensor) -> torch.Tensor:
    """Keep per-channel decoder inputs near the normalized V-JEPA distribution."""
    values = features.float()
    if values.ndim == 5:
        reduction_dims = (0, 1, 3, 4)
    elif values.ndim == 4:
        reduction_dims = (0, 2, 3)
    else:
        raise ValueError(f"Expected 4D or 5D feature tensor, got {tuple(values.shape)}")
    mean = values.mean(dim=reduction_dims)
    std = values.var(dim=reduction_dims, unbiased=False).add(1.0e-6).sqrt()
    return mean.square().mean() + (std - 1.0).square().mean()


def image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    delta = prediction.float() - target.float()
    l1 = delta.abs().flatten(1).mean(1)
    mse = delta.square().flatten(1).mean(1).clamp_min(1.0e-10)
    psnr = -10.0 * torch.log10(mse)
    return {"l1": l1, "mse": mse, "psnr": psnr}


def latent_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    pred = prediction.float().flatten(1)
    true = target.float().flatten(1)
    return {
        "l1": (pred - true).abs().mean(1),
        "cosine": 1.0 - torch.nn.functional.cosine_similarity(pred, true, dim=1),
    }
