"""Frozen official V-JEPA 2.1 predictor and JEPA-WMs decoder interfaces."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.vjepa21_jepawms.common import project_path, volume_from_tokens
from experiments.vjepa21_jepawms.imports import get_official_decoder_class


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


class OfficialVJEPA21WorldModel:
    """Exact Meta V-JEPA 2.1 ViT-g encoder/predictor, frozen for inference."""

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
        if self.future_slots != 1:
            raise ValueError(
                "The two-frame experiment requires exactly one held-out JEPA tubelet"
            )

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

    def _normalize_levels(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = self.levels * self.embed_dim
        if tokens.shape[-1] != expected:
            raise ValueError(f"Expected {expected} feature channels, got {tokens.shape[-1]}")
        shape = tokens.shape
        levels = tokens.reshape(*shape[:-1], self.levels, self.embed_dim)
        levels = F.layer_norm(levels.float(), (self.embed_dim,))
        return levels.reshape(*shape[:-1], expected).to(torch.bfloat16)

    @torch.inference_mode()
    def target_tubelet(self, normalized_video: torch.Tensor) -> torch.Tensor:
        pixel = normalized_video.to(self.device, dtype=torch.bfloat16)
        pixel = pixel.permute(0, 2, 1, 3, 4).contiguous()
        tokens = self._normalize_levels(self.encoder(pixel))
        target = tokens[:, -self.spatial_tokens :]
        return volume_from_tokens(target, 1, self.grid_height, self.grid_width)

    @torch.inference_mode()
    def predict_tubelet(self, normalized_context: torch.Tensor) -> torch.Tensor:
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
        prediction = self._normalize_levels(prediction)
        return volume_from_tokens(
            prediction, 1, self.grid_height, self.grid_width
        )


class FrozenJEPAWMSImageDecoder:
    """The exact upstream ``VisionTransformerDecoder`` with frozen weights."""

    def __init__(self, config: dict, device: torch.device) -> None:
        decoder_cfg = config["jepa_wms_decoder"]
        dependency_root = project_path(decoder_cfg["dependency_root"])
        decoder_class = get_official_decoder_class(dependency_root)
        architecture = dict(decoder_cfg["architecture"])
        self.image_size = int(architecture["img_size"][0])
        self.patch_size = int(architecture["patch_size"])
        self.decoder = decoder_class(**architecture)

        checkpoint_path = project_path(decoder_cfg["checkpoint_path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"JEPA-WMs decoder checkpoint not found: {checkpoint_path}. Run assets first."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" not in checkpoint:
            raise KeyError(f"Decoder checkpoint has no 'model' key: {list(checkpoint)[:12]}")
        state = {
            key.replace("module.", ""): value
            for key, value in checkpoint["model"].items()
            if key.replace("module.", "") != "decoder_pos_embed"
        }
        message = self.decoder.load_state_dict(state, strict=False)
        unexpected = list(message.unexpected_keys)
        missing = [key for key in message.missing_keys if key != "decoder_pos_embed"]
        if unexpected or missing:
            raise RuntimeError(
                f"JEPA-WMs decoder mismatch; missing={missing}, unexpected={unexpected}"
            )
        self.decoder.requires_grad_(False).to(device=device, dtype=torch.bfloat16).eval()
        self.device = device
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    def train(self, mode: bool = True):
        self.decoder.eval()
        return self

    def decode_normalized(self, features: torch.Tensor) -> torch.Tensor:
        """Decode ``(B,T,1408,H,W)`` features to ImageNet pixels ``(B,T,3,H,W)``."""
        if features.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(features.shape)}")
        feature_grid = features.permute(0, 1, 3, 4, 2).unsqueeze(2)
        patches = self.decoder(feature_grid)
        batch, frames, views, height, width, channels = patches.shape
        if views != 1:
            raise ValueError(f"Expected one decoder view, got {views}")
        expected = self.patch_size * self.patch_size * 3
        if channels != expected:
            raise ValueError(f"Decoder patch channels {channels} != {expected}")
        pixels = (
            patches[:, :, 0]
            .reshape(batch, frames, height, width, self.patch_size, self.patch_size, 3)
            .permute(0, 1, 6, 2, 4, 3, 5)
            .reshape(batch, frames, 3, height * self.patch_size, width * self.patch_size)
        )
        return pixels.float()

    def decode_rgb01(self, features: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        normalized = self.decode_normalized(features)
        rgb = normalized * self.std + self.mean
        return rgb.clamp(0.0, 1.0) if clamp else rgb


def validate_geometry(config: dict) -> None:
    data = config["data"]
    model = config["vjepa21"]
    adapter = config["adapter"]
    decoder = config["jepa_wms_decoder"]["architecture"]
    if data["num_frames"] != 16 or data["context_frames"] != 14:
        raise ValueError("This experiment is defined as 14 context + 2 held-out frames")
    if data["num_frames"] % model["tubelet_size"] != 0:
        raise ValueError("num_frames must be divisible by the V-JEPA tubelet size")
    if data["context_frames"] % model["tubelet_size"] != 0:
        raise ValueError("context_frames must be divisible by the V-JEPA tubelet size")
    if data.get("target_frame_indices") != [14, 15]:
        raise ValueError("target_frame_indices must be [14, 15] for held-out frames 15-16")
    if model["feature_levels"] * model["embed_dim"] != adapter["input_dim"]:
        raise ValueError("Adapter input_dim must contain all V-JEPA 2.1 feature levels")
    if adapter["output_dim"] != decoder["embed_dim"]:
        raise ValueError("Adapter output_dim must equal the JEPA-WMs decoder embed_dim")
    if adapter.get("output_frames") != model["tubelet_size"]:
        raise ValueError("Adapter output_frames must equal the V-JEPA tubelet size")
    if decoder["img_size"][0] != decoder["img_size"][1]:
        raise ValueError("Only a square decoder output is supported")
