"""Trainable adapter from FactorJEPA token volumes to Cosmos-CV latents."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return inputs + hidden


class JEPAToCosmosAdapter(nn.Module):
    """A small spatiotemporal mapper with an explicit target latent grid."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_channels: int,
        residual_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *(ResidualBlock3D(hidden_dim, dropout) for _ in range(residual_blocks))
        )
        self.output_norm = nn.GroupNorm(min(32, hidden_dim), hidden_dim)
        self.output_projection = nn.Conv3d(hidden_dim, output_channels, kernel_size=3, padding=1)

    def forward(
        self,
        jepa_volume: torch.Tensor,
        target_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        if jepa_volume.ndim != 5:
            raise ValueError(f"Expected (B,D,T,H,W), got {tuple(jepa_volume.shape)}")
        hidden = jepa_volume.permute(0, 2, 3, 4, 1)
        hidden = self.input_projection(self.input_norm(hidden))
        hidden = hidden.permute(0, 4, 1, 2, 3).contiguous()
        hidden = self.blocks(hidden)
        hidden = F.interpolate(hidden, size=target_shape, mode="trilinear", align_corners=False)
        return self.output_projection(F.silu(self.output_norm(hidden)))


def build_adapter(config: dict) -> JEPAToCosmosAdapter:
    return JEPAToCosmosAdapter(
        input_dim=config["adapter"]["input_dim"],
        hidden_dim=config["adapter"]["hidden_dim"],
        output_channels=config["cosmos"]["latent_channels"],
        residual_blocks=config["adapter"]["residual_blocks"],
        dropout=config["adapter"]["dropout"],
    )

