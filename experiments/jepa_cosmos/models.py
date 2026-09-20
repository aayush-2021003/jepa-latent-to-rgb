"""Frozen FactorJEPA and Cosmos tokenizer interfaces used by the experiment."""
from __future__ import annotations

import sys

import torch

from experiments.jepa_cosmos.common import PROJECT_ROOT, project_path, volume_from_tokens


SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FactorJEPAWorldModel:
    """Causal half-clip prediction over a frozen FactorJEPA encoder/predictor."""

    def __init__(self, config: dict, device: torch.device) -> None:
        from utils.predictor_eval import load_encoder_predictor

        data_cfg = config["data"]
        factor_cfg = config["factorjepa"]
        model_cfg_path = project_path(factor_cfg["model_config"])
        checkpoint_path = project_path(factor_cfg["checkpoint_path"])
        self.encoder, self.predictor, _ = load_encoder_predictor(
            checkpoint_path,
            data_cfg["num_frames"],
            str(model_cfg_path),
        )
        self.encoder.requires_grad_(False)
        self.predictor.requires_grad_(False)
        self.encoder.eval()
        self.predictor.eval()
        self.device = device

        import yaml
        with model_cfg_path.open() as handle:
            model_cfg = yaml.safe_load(handle)["model"]
        self.embed_dim = model_cfg["embed_dim"]
        if config["adapter"]["input_dim"] != self.embed_dim:
            raise ValueError(
                "Adapter input_dim does not match the selected FactorJEPA model: "
                f"{config['adapter']['input_dim']} != {self.embed_dim}"
            )
        self.patch_size = model_cfg["patch_size"]
        self.tubelet_size = model_cfg["tubelet_size"]
        self.crop_size = model_cfg["crop_size"]
        self.num_frames = data_cfg["num_frames"]
        self.context_frames = data_cfg["context_frames"]
        if self.context_frames % self.tubelet_size != 0:
            raise ValueError("context_frames must be divisible by the JEPA tubelet size")
        if self.num_frames % self.tubelet_size != 0:
            raise ValueError("num_frames must be divisible by the JEPA tubelet size")
        self.total_slots = self.num_frames // self.tubelet_size
        self.context_slots = self.context_frames // self.tubelet_size
        self.future_slots = self.total_slots - self.context_slots
        self.grid_height = self.crop_size // self.patch_size
        self.grid_width = self.crop_size // self.patch_size
        self.spatial_tokens = self.grid_height * self.grid_width

    def _indices(self, first_slot: int, last_slot: int, batch_size: int) -> torch.Tensor:
        slots = torch.arange(first_slot, last_slot, device=self.device)
        spatial = torch.arange(self.spatial_tokens, device=self.device)
        indices = (slots[:, None] * self.spatial_tokens + spatial[None, :]).reshape(-1)
        return indices.unsqueeze(0).expand(batch_size, -1).contiguous()

    @staticmethod
    def _concat_hierarchical(output: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(output, (list, tuple)):
            return torch.cat(list(output), dim=-1)
        return output

    def _final_layer(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] < self.embed_dim:
            raise ValueError(f"JEPA output dim {tokens.shape[-1]} < embed_dim {self.embed_dim}")
        return tokens[..., -self.embed_dim:]

    @torch.no_grad()
    def target_future(self, normalized_video: torch.Tensor) -> torch.Tensor:
        """Encode true future frames; input is ``(B,T,3,H,W)``."""
        pixel = normalized_video.to(self.device, dtype=torch.bfloat16)
        pixel = pixel.permute(0, 2, 1, 3, 4).contiguous()
        tokens = self._final_layer(self._concat_hierarchical(self.encoder(pixel)))
        start = self.context_slots * self.spatial_tokens
        future = tokens[:, start:, :]
        return volume_from_tokens(
            future,
            self.future_slots,
            self.grid_height,
            self.grid_width,
        )

    @torch.no_grad()
    def predict_future(self, normalized_context: torch.Tensor) -> torch.Tensor:
        """Predict future tokens from context only; no future pixels are consumed."""
        if normalized_context.shape[1] != self.context_frames:
            raise ValueError(
                f"Expected {self.context_frames} context frames, got {normalized_context.shape[1]}"
            )
        batch_size = normalized_context.shape[0]
        blank_future = torch.zeros(
            batch_size,
            self.num_frames - self.context_frames,
            *normalized_context.shape[2:],
            dtype=normalized_context.dtype,
            device=normalized_context.device,
        )
        padded = torch.cat((normalized_context, blank_future), dim=1)
        pixel = padded.to(self.device, dtype=torch.bfloat16).permute(0, 2, 1, 3, 4).contiguous()
        mask_context = self._indices(0, self.context_slots, batch_size)
        mask_future = self._indices(self.context_slots, self.total_slots, batch_size)
        context_tokens = self._concat_hierarchical(self.encoder(pixel, masks=[mask_context]))
        output = self.predictor(
            context_tokens,
            [mask_context],
            [mask_future],
            mask_index=0,
        )
        if isinstance(output, tuple):
            output = output[0]
        output = self._final_layer(output)
        return volume_from_tokens(
            output,
            self.future_slots,
            self.grid_height,
            self.grid_width,
        )


class CosmosContinuousTokenizer:
    """Official Cosmos tokenizer with strict anchored ``4k+1`` temporal handling."""

    def __init__(
        self,
        config: dict,
        device: torch.device,
        load_encoder: bool,
        load_decoder: bool,
    ) -> None:
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer

        cosmos_cfg = config["cosmos"]
        checkpoint_dir = project_path(cosmos_cfg["checkpoint_dir"])
        encoder_path = checkpoint_dir / cosmos_cfg["encoder_file"]
        decoder_path = checkpoint_dir / cosmos_cfg["decoder_file"]
        if load_encoder and not encoder_path.is_file():
            raise FileNotFoundError(f"Cosmos encoder not found: {encoder_path}")
        if load_decoder and not decoder_path.is_file():
            raise FileNotFoundError(f"Cosmos decoder not found: {decoder_path}")
        self.temporal_compression = cosmos_cfg["temporal_compression"]
        self.dtype = getattr(torch, cosmos_cfg["dtype"])
        self.device = device
        self.model = CausalVideoTokenizer(
            checkpoint_enc=str(encoder_path) if load_encoder else None,
            checkpoint_dec=str(decoder_path) if load_decoder else None,
            device=str(device),
            dtype=cosmos_cfg["dtype"],
        )
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a strict ``4k+1`` sequence without silently repeating frames."""
        frames = video.shape[2]
        if (frames - 1) % self.temporal_compression != 0:
            raise ValueError(
                f"Cosmos input must contain 4k+1 frames; received {frames}. "
                "Use the last observed context frame as the causal anchor."
            )
        output = self.model.encode(video.to(self.device, dtype=self.dtype))
        if isinstance(output, tuple):
            output = output[0]
        return output

    def _anchored_latent(
        self,
        anchor: torch.Tensor,
        future: torch.Tensor,
        future_frames: int,
    ) -> torch.Tensor:
        if anchor.shape[2] != 1:
            raise ValueError(f"Expected one anchor latent slot, got {anchor.shape[2]}")
        if anchor.shape[:2] != future.shape[:2] or anchor.shape[-2:] != future.shape[-2:]:
            raise ValueError(
                f"Anchor/future latent geometry mismatch: {anchor.shape} vs {future.shape}"
            )
        expected_future_slots = future_frames // self.temporal_compression
        if future_frames % self.temporal_compression != 0:
            raise ValueError("Anchored future length must be divisible by temporal compression")
        if future.shape[2] != expected_future_slots:
            raise ValueError(
                f"Expected {expected_future_slots} future latent slots, got {future.shape[2]}"
            )
        anchor = anchor.to(device=future.device, dtype=future.dtype)
        return torch.cat((anchor, future), dim=2)

    def decode_anchored(
        self,
        anchor: torch.Tensor,
        future: torch.Tensor,
        future_frames: int,
    ) -> torch.Tensor:
        """Decode a known context anchor plus future latents, then remove the anchor frame."""
        latent = self._anchored_latent(anchor, future, future_frames)
        decoded = self.model.decode(latent.to(self.device, dtype=self.dtype))
        return decoded[:, :, 1:future_frames + 1]

    def decode_anchored_with_input_grad(
        self,
        anchor: torch.Tensor,
        future: torch.Tensor,
        future_frames: int,
    ) -> torch.Tensor:
        """Anchored decoding with gradients only into the predicted future latent."""
        latent = self._anchored_latent(anchor, future, future_frames)
        decoder = self.model._dec_model
        if decoder is None:
            raise RuntimeError("Cosmos decoder was not loaded")
        decoded = decoder(latent.to(self.device, dtype=self.dtype))
        return decoded[:, :, 1:future_frames + 1]


def validate_model_geometry(config: dict) -> None:
    data_cfg = config["data"]
    model_cfg = config["factorjepa"]
    import yaml

    model_config_path = project_path(model_cfg["model_config"])
    if not model_config_path.is_file():
        raise FileNotFoundError(f"FactorJEPA model config not found: {model_config_path}")
    with model_config_path.open() as handle:
        architecture = yaml.safe_load(handle)["model"]
    if data_cfg["context_frames"] >= data_cfg["num_frames"]:
        raise ValueError("context_frames must be smaller than num_frames")
    if data_cfg["crop_size"] != architecture["crop_size"]:
        raise ValueError(
            f"Data crop_size {data_cfg['crop_size']} does not match FactorJEPA "
            f"crop_size {architecture['crop_size']}"
        )
    if config["adapter"]["input_dim"] != architecture["embed_dim"]:
        raise ValueError(
            f"Adapter input_dim {config['adapter']['input_dim']} does not match "
            f"FactorJEPA embed_dim {architecture['embed_dim']}"
        )
    if data_cfg["num_frames"] % architecture["tubelet_size"] != 0:
        raise ValueError("num_frames must be divisible by the FactorJEPA tubelet size")
    if data_cfg["context_frames"] % architecture["tubelet_size"] != 0:
        raise ValueError("context_frames must be divisible by the FactorJEPA tubelet size")
    if data_cfg["crop_size"] % 8 != 0:
        raise ValueError("crop_size must be divisible by Cosmos' spatial compression factor 8")
    future_frames = data_cfg["num_frames"] - data_cfg["context_frames"]
    temporal_compression = config["cosmos"]["temporal_compression"]
    if config["cosmos"].get("anchor_last_context_frame") is not True:
        raise ValueError("Set cosmos.anchor_last_context_frame=true for leakage-safe 4k+1 decoding")
    if future_frames % temporal_compression != 0:
        raise ValueError(
            "With a context anchor, the future frame count must be divisible by the "
            "Cosmos temporal compression factor"
        )
    if model_cfg["use_final_layer_only"] is not True:
        raise ValueError(
            "This implementation intentionally uses the predictor's final-layer slice; "
            "set factorjepa.use_final_layer_only=true."
        )
