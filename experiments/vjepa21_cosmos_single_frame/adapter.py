"""Spatial adapter from one predicted V-JEPA tubelet to one Cosmos-CI latent."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = min(32, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return inputs + hidden


class VJEPA21ToCosmosImageAdapter(nn.Module):
    """Map ``(B,D,1,H,W)`` to a single ``(B,16,h,w)`` image latent."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_channels: int,
        residual_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *(ResidualBlock2D(hidden_dim, dropout) for _ in range(residual_blocks))
        )
        groups = min(32, hidden_dim)
        while hidden_dim % groups:
            groups -= 1
        self.output_norm = nn.GroupNorm(groups, hidden_dim)
        self.output_projection = nn.Conv2d(
            hidden_dim, output_channels, kernel_size=3, padding=1
        )

    def forward(
        self, features: torch.Tensor, target_shape: tuple[int, int]
    ) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(f"Expected (B,D,T,H,W), got {tuple(features.shape)}")
        if features.shape[1] != self.input_dim or features.shape[2] != 1:
            raise ValueError(
                f"Expected D={self.input_dim}, T=1; got {tuple(features.shape)}"
            )
        if len(target_shape) != 2:
            raise ValueError(f"Expected a spatial target shape, got {target_shape}")
        hidden = features[:, :, 0].permute(0, 2, 3, 1)
        hidden = self.input_projection(self.input_norm(hidden))
        hidden = hidden.permute(0, 3, 1, 2).contiguous()
        hidden = self.blocks(hidden)
        hidden = F.interpolate(
            hidden, size=target_shape, mode="bilinear", align_corners=False
        )
        return self.output_projection(F.silu(self.output_norm(hidden)))


def build_adapter(config: dict) -> VJEPA21ToCosmosImageAdapter:
    adapter = config["adapter"]
    return VJEPA21ToCosmosImageAdapter(
        input_dim=int(adapter["input_dim"]),
        hidden_dim=int(adapter["hidden_dim"]),
        output_channels=int(config["cosmos"]["latent_channels"]),
        residual_blocks=int(adapter["residual_blocks"]),
        dropout=float(adapter["dropout"]),
    )
