"""High-capacity bridge from four V-JEPA 2.1 levels to JEPA-WMs features."""
from __future__ import annotations

import torch
from torch import nn


class SpatialResidualBlock(nn.Module):
    """ConvNeXt-style spatial block with no channel bottleneck at its output."""

    def __init__(self, channels: int, expansion: int, drop: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        hidden = channels * expansion
        self.expand = nn.Linear(channels, hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(drop)
        self.contract = nn.Linear(hidden, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1.0e-4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.depthwise(inputs).permute(0, 2, 3, 1)
        hidden = self.norm(hidden)
        hidden = self.contract(self.dropout(self.activation(self.expand(hidden))))
        hidden = (hidden * self.layer_scale).permute(0, 3, 1, 2)
        return residual + hidden


class VJEPA21ToJEPAWMSAdapter(nn.Module):
    """Fuse all deep-supervision levels and preserve the decoder's width."""

    def __init__(
        self,
        levels: int = 4,
        level_dim: int = 1408,
        output_dim: int = 1408,
        residual_blocks: int = 6,
        expansion: int = 2,
        dropout: float = 0.0,
        output_frames: int = 2,
    ) -> None:
        super().__init__()
        if levels <= 0 or level_dim <= 0 or output_dim <= 0:
            raise ValueError("Adapter dimensions must be positive")
        self.levels = levels
        self.level_dim = level_dim
        self.input_dim = levels * level_dim
        self.output_dim = output_dim
        self.output_frames = output_frames
        if output_frames <= 0:
            raise ValueError("output_frames must be positive")
        self.fusion = nn.Conv2d(self.input_dim, output_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                SpatialResidualBlock(output_dim, expansion, dropout)
                for _ in range(residual_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(output_dim, eps=1e-6)
        self.frame_heads = nn.ModuleList(
            [nn.Conv2d(output_dim, output_dim, kernel_size=1) for _ in range(output_frames)]
        )
        self._initialize_fusion()
        self._initialize_frame_heads()

    def _initialize_fusion(self) -> None:
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        shared = min(self.level_dim, self.output_dim)
        with torch.no_grad():
            for level in range(self.levels):
                indices = torch.arange(shared)
                self.fusion.weight[indices, level * self.level_dim + indices, 0, 0] = (
                    1.0 / self.levels
                )

    def _initialize_frame_heads(self) -> None:
        # Both heads initially see the complete shared representation. Training
        # can then specialize one head for each frame in the predicted tubelet.
        with torch.no_grad():
            for head in self.frame_heads:
                nn.init.dirac_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``(B,5632,1,H,W)`` to ``(B,2,1408,H,W)``."""
        if features.ndim != 5:
            raise ValueError(f"Expected (B,C,T,H,W), got {tuple(features.shape)}")
        if features.shape[1] != self.input_dim or features.shape[2] != 1:
            raise ValueError(
                f"Expected C={self.input_dim}, T=1; got {tuple(features.shape)}"
            )
        hidden = self.blocks(self.fusion(features[:, :, 0]))
        hidden = self.output_norm(hidden.permute(0, 2, 3, 1))
        hidden = hidden.permute(0, 3, 1, 2).contiguous()
        return torch.stack([head(hidden) for head in self.frame_heads], dim=1)
