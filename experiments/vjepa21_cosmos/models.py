"""Frozen official V-JEPA 2.1 predictor and Cosmos tokenizer interfaces."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.vjepa21_cosmos.common import project_path, volume_from_tokens
from experiments.jepa_cosmos.models import CosmosContinuousTokenizer


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


class OfficialVJEPA21WorldModel:
    """Official ViT-g/16 1B target encoder and predictor, frozen for inference."""

    def __init__(self, config: dict, device: torch.device) -> None:
        from utils.vjepa2_imports import (
            get_vit_giant_xformers_2_1,
            get_vit_predictor_2_1,
        )

        data_cfg = config["data"]
        model_cfg = config["vjepa21"]
        self.device = device
        self.num_frames = int(data_cfg["num_frames"])
        self.context_frames = int(data_cfg["context_frames"])
        self.crop_size = int(data_cfg["crop_size"])
        self.patch_size = int(model_cfg["patch_size"])
        self.tubelet_size = int(model_cfg["tubelet_size"])
        self.embed_dim = int(model_cfg["embed_dim"])
        self.levels = int(model_cfg["feature_levels"])
        self.grid_height = self.crop_size // self.patch_size
        self.grid_width = self.grid_height
        self.spatial_tokens = self.grid_height * self.grid_width
        self.total_slots = self.num_frames // self.tubelet_size
        self.context_slots = self.context_frames // self.tubelet_size
        self.future_slots = self.total_slots - self.context_slots

        encoder_ctor = get_vit_giant_xformers_2_1()
        predictor_ctor = get_vit_predictor_2_1()
        shared = dict(
            img_size=(self.crop_size, self.crop_size),
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        self.encoder = encoder_ctor(**shared)
        self.encoder.return_hierarchical = True
        self.predictor = predictor_ctor(
            img_size=(self.crop_size, self.crop_size),
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            embed_dim=self.embed_dim,
            predictor_embed_dim=int(model_cfg["predictor_embed_dim"]),
            depth=int(model_cfg["predictor_depth"]),
            num_heads=int(model_cfg["predictor_heads"]),
            use_mask_tokens=True,
            num_mask_tokens=int(model_cfg["predictor_mask_tokens"]),
            zero_init_mask_tokens=True,
            use_rope=True,
            uniform_power=False,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            n_output_distillation=self.levels,
            return_all_tokens=True,
            img_temporal_dim_size=1,
        )

        checkpoint_path = project_path(model_cfg["checkpoint_path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"V-JEPA 2.1 checkpoint not found: {checkpoint_path}. Run assets first."
            )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        encoder_key = model_cfg.get("encoder_checkpoint_key", "target_encoder")
        if encoder_key not in checkpoint or "predictor" not in checkpoint:
            raise KeyError(
                f"Checkpoint must contain {encoder_key!r} and 'predictor'; "
                f"found {list(checkpoint)[:12]}"
            )
        self.encoder.load_state_dict(
            _clean_state_dict(checkpoint[encoder_key]), strict=True
        )
        self.predictor.load_state_dict(
            _clean_state_dict(checkpoint["predictor"]), strict=True
        )
        del checkpoint
        self.encoder.requires_grad_(False).to(device=device, dtype=torch.bfloat16).eval()
        self.predictor.requires_grad_(False).to(device=device, dtype=torch.bfloat16).eval()

    def _indices(self, first_slot: int, last_slot: int, batch: int) -> torch.Tensor:
        slots = torch.arange(first_slot, last_slot, device=self.device)
        spatial = torch.arange(self.spatial_tokens, device=self.device)
        indices = (slots[:, None] * self.spatial_tokens + spatial[None, :]).reshape(-1)
        return indices.unsqueeze(0).expand(batch, -1).contiguous()

    def _final_level(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = self.levels * self.embed_dim
        if tokens.shape[-1] != expected:
            raise ValueError(f"Expected {expected} hierarchical channels, got {tokens.shape[-1]}")
        final = tokens.reshape(*tokens.shape[:-1], self.levels, self.embed_dim)[..., -1, :]
        return F.layer_norm(final.float(), (self.embed_dim,)).to(torch.bfloat16)

    @torch.inference_mode()
    def target_future(self, normalized_video: torch.Tensor) -> torch.Tensor:
        pixel = normalized_video.to(self.device, dtype=torch.bfloat16)
        pixel = pixel.permute(0, 2, 1, 3, 4).contiguous()
        tokens = self._final_level(self.encoder(pixel))
        start = self.context_slots * self.spatial_tokens
        return volume_from_tokens(
            tokens[:, start:], self.future_slots, self.grid_height, self.grid_width
        )

    @torch.inference_mode()
    def predict_future(self, normalized_context: torch.Tensor) -> torch.Tensor:
        if normalized_context.shape[1] != self.context_frames:
            raise ValueError(
                f"Expected {self.context_frames} context frames, got "
                f"{normalized_context.shape[1]}"
            )
        batch = normalized_context.shape[0]
        blank = torch.zeros(
            batch,
            self.num_frames - self.context_frames,
            *normalized_context.shape[2:],
            dtype=normalized_context.dtype,
            device=normalized_context.device,
        )
        pixel = torch.cat((normalized_context, blank), dim=1)
        pixel = pixel.to(self.device, dtype=torch.bfloat16).permute(0, 2, 1, 3, 4)
        context_mask = self._indices(0, self.context_slots, batch)
        target_mask = self._indices(self.context_slots, self.total_slots, batch)
        context = self.encoder(pixel.contiguous(), masks=[context_mask])
        prediction, _ = self.predictor(
            context, [context_mask], [target_mask], mask_index=0
        )
        prediction = self._final_level(prediction)
        return volume_from_tokens(
            prediction, self.future_slots, self.grid_height, self.grid_width
        )


def validate_model_geometry(config: dict) -> None:
    data = config["data"]
    model = config["vjepa21"]
    cosmos = config["cosmos"]
    adapter = config["adapter"]
    if data["num_frames"] != 16 or data["context_frames"] != 12:
        raise ValueError("This experiment requires 12 context + 4 future frames")
    if data["crop_size"] % model["patch_size"] != 0:
        raise ValueError("crop_size must be divisible by the V-JEPA patch size")
    if data["crop_size"] % 8 != 0:
        raise ValueError("crop_size must be divisible by Cosmos spatial compression 8")
    if data["num_frames"] % model["tubelet_size"] != 0:
        raise ValueError("num_frames must be divisible by the V-JEPA tubelet size")
    if data["context_frames"] % model["tubelet_size"] != 0:
        raise ValueError("context_frames must be divisible by the V-JEPA tubelet size")
    future_frames = data["num_frames"] - data["context_frames"]
    if future_frames != 4 or future_frames % cosmos["temporal_compression"] != 0:
        raise ValueError("Cosmos CV4 requires exactly four future frames here")
    if cosmos.get("anchor_last_context_frame") is not True:
        raise ValueError("cosmos.anchor_last_context_frame must be true")
    if adapter["input_dim"] != model["embed_dim"]:
        raise ValueError("Adapter input_dim must equal the final V-JEPA level width")
    if model.get("use_final_layer_only") is not True:
        raise ValueError("Set vjepa21.use_final_layer_only=true for an exact prior-loss repeat")
    if cosmos["latent_channels"] != adapter.get("output_channels", 16):
        raise ValueError("Adapter output_channels must equal Cosmos latent_channels")
