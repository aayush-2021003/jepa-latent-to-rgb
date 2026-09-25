"""Frozen V-JEPA 2.1 predictor and Cosmos continuous-image tokenizer."""
from __future__ import annotations

import torch

from experiments.vjepa21_cosmos.models import OfficialVJEPA21WorldModel
from experiments.vjepa21_cosmos_single_frame.common import project_path


class CosmosContinuousImageTokenizer:
    """Thin, shape-checked wrapper around the official Cosmos CI tokenizer."""

    def __init__(
        self,
        config: dict,
        device: torch.device,
        load_encoder: bool,
        load_decoder: bool,
    ) -> None:
        from cosmos_tokenizer.image_lib import ImageTokenizer

        cosmos = config["cosmos"]
        checkpoint_dir = project_path(cosmos["checkpoint_dir"])
        encoder_path = checkpoint_dir / cosmos["encoder_file"]
        decoder_path = checkpoint_dir / cosmos["decoder_file"]
        if load_encoder and not encoder_path.is_file():
            raise FileNotFoundError(f"Cosmos image encoder not found: {encoder_path}")
        if load_decoder and not decoder_path.is_file():
            raise FileNotFoundError(f"Cosmos image decoder not found: {decoder_path}")
        self.device = device
        self.dtype = getattr(torch, cosmos["dtype"])
        self.model = ImageTokenizer(
            checkpoint_enc=str(encoder_path) if load_encoder else None,
            checkpoint_dec=str(decoder_path) if load_decoder else None,
            device=str(device),
            dtype=cosmos["dtype"],
        )
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape (B,3,H,W), got {tuple(image.shape)}")
        output = self.model.encode(image.to(self.device, dtype=self.dtype))
        if isinstance(output, tuple):
            output = output[0]
        if output.ndim != 4:
            raise RuntimeError(f"Unexpected Cosmos-CI latent shape: {tuple(output.shape)}")
        return output

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(f"Expected latent shape (B,C,H,W), got {tuple(latent.shape)}")
        output = self.model.decode(latent.to(self.device, dtype=self.dtype))
        return output[0] if isinstance(output, tuple) else output


def validate_model_geometry(config: dict) -> None:
    data = config["data"]
    model = config["vjepa21"]
    cosmos = config["cosmos"]
    adapter = config["adapter"]
    if int(data["num_frames"]) != 16 or int(data["context_frames"]) != 14:
        raise ValueError("Single-frame readout requires 14 context + 2 held-out frames")
    if int(data.get("target_frame_offset", 0)) != 0:
        raise ValueError("target_frame_offset must be 0 to supervise the earliest frame 15")
    if int(model["tubelet_size"]) != 2:
        raise ValueError("This experiment is defined for V-JEPA's two-frame tubelets")
    if data["crop_size"] % model["patch_size"]:
        raise ValueError("crop_size must be divisible by the V-JEPA patch size")
    if data["crop_size"] % cosmos["spatial_compression"]:
        raise ValueError("crop_size must be divisible by Cosmos spatial compression")
    if data["context_frames"] % model["tubelet_size"]:
        raise ValueError("context_frames must end at a complete V-JEPA tubelet")
    if data["num_frames"] - data["context_frames"] != model["tubelet_size"]:
        raise ValueError("Exactly one two-frame V-JEPA tubelet must be held out")
    if adapter["input_dim"] != model["embed_dim"]:
        raise ValueError("Adapter input_dim must match V-JEPA final-level width")
    if model.get("use_final_layer_only") is not True:
        raise ValueError("Set vjepa21.use_final_layer_only=true")
    if cosmos["latent_channels"] != adapter["output_channels"]:
        raise ValueError("Adapter output_channels must match Cosmos latent_channels")
