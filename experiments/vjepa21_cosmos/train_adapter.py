"""Predicted-only training with four-frame decoded validation metrics."""
from __future__ import annotations

import torch
from tqdm import tqdm

from experiments.jepa_cosmos import train_adapter as implementation
from experiments.jepa_cosmos.adapter import build_adapter
from experiments.jepa_cosmos.losses import latent_alignment_loss
from experiments.jepa_cosmos.models import CosmosContinuousTokenizer
from experiments.vjepa21_cosmos.common import load_config
from experiments.vjepa21_cosmos.data import LatentShardDataset, cache_collate
from experiments.vjepa21_cosmos.models import validate_model_geometry
from experiments.vjepa21_cosmos.tracking import push_checkpoint_to_hub, start_wandb


_validation_decoder = None


class RecordingCosmosTokenizer(CosmosContinuousTokenizer):
    """Expose the training loop's decoder to the validation metric function."""

    def __init__(self, config, device, load_encoder, load_decoder):
        super().__init__(config, device, load_encoder, load_decoder)
        if load_decoder:
            global _validation_decoder
            _validation_decoder = self


@torch.no_grad()
def validate(adapter, loader, device, config) -> dict[str, float]:
    if _validation_decoder is None:
        raise RuntimeError(
            "Decoded validation requires tracking.log_validation_videos=true"
        )
    adapter.eval()
    loss_cfg = config["loss"]
    totals = {
        "oracle_loss": 0.0,
        "oracle_l1": 0.0,
        "oracle_cosine": 0.0,
        "predicted_loss": 0.0,
        "predicted_l1": 0.0,
        "predicted_cosine": 0.0,
        "predicted_rgb_l1": 0.0,
        "predicted_rgb_psnr": 0.0,
    }
    for frame_number in range(13, 17):
        totals[f"frame{frame_number}_predicted_rgb_l1"] = 0.0
        totals[f"frame{frame_number}_predicted_rgb_psnr"] = 0.0
    samples = 0
    precision = implementation.dtype_from_name(config["training"]["mixed_precision"])
    future_frames = config["data"]["num_frames"] - config["data"]["context_frames"]
    for batch in tqdm(loader, desc="Validating", unit="batch", leave=False):
        target_jepa = batch["jepa_target"].to(device, non_blocking=True)
        predicted_jepa = batch["jepa_predicted"].to(device, non_blocking=True)
        cosmos_anchor = batch["cosmos_anchor"].to(device, non_blocking=True)
        cosmos_target = batch["cosmos_target"].to(device, non_blocking=True)
        target_rgb = (
            batch["future_rgb"]
            .to(device, non_blocking=True)
            .float()
            .permute(0, 2, 1, 3, 4)
            / 255.0
        )
        target_shape = tuple(cosmos_target.shape[-3:])
        with torch.autocast("cuda", dtype=precision):
            oracle_latent = adapter(target_jepa, target_shape)
            predicted_latent = adapter(predicted_jepa, target_shape)
        oracle_loss, oracle_parts = latent_alignment_loss(
            oracle_latent,
            cosmos_target,
            loss_cfg["latent_l1"],
            loss_cfg["latent_cosine"],
        )
        predicted_loss, predicted_parts = latent_alignment_loss(
            predicted_latent,
            cosmos_target,
            loss_cfg["latent_l1"],
            loss_cfg["latent_cosine"],
        )
        decoded = _validation_decoder.decode_anchored(
            cosmos_anchor, predicted_latent, future_frames
        )
        predicted_rgb = (decoded.float().clamp(-1.0, 1.0) + 1.0) / 2.0
        delta = predicted_rgb - target_rgb
        frame_l1 = delta.abs().mean(dim=(1, 3, 4))
        frame_mse = delta.square().mean(dim=(1, 3, 4)).clamp_min(1.0e-10)
        frame_psnr = -10.0 * torch.log10(frame_mse)
        batch_size = target_jepa.shape[0]
        samples += batch_size
        totals["oracle_loss"] += float(oracle_loss) * batch_size
        totals["oracle_l1"] += float(oracle_parts["latent_l1"]) * batch_size
        totals["oracle_cosine"] += float(oracle_parts["latent_cosine"]) * batch_size
        totals["predicted_loss"] += float(predicted_loss) * batch_size
        totals["predicted_l1"] += float(predicted_parts["latent_l1"]) * batch_size
        totals["predicted_cosine"] += float(predicted_parts["latent_cosine"]) * batch_size
        totals["predicted_rgb_l1"] += frame_l1.mean(1).sum().item()
        totals["predicted_rgb_psnr"] += frame_psnr.mean(1).sum().item()
        for frame_index, frame_number in enumerate(range(13, 17)):
            totals[f"frame{frame_number}_predicted_rgb_l1"] += frame_l1[:, frame_index].sum().item()
            totals[f"frame{frame_number}_predicted_rgb_psnr"] += frame_psnr[:, frame_index].sum().item()
    if samples == 0:
        raise RuntimeError("Validation cache yielded zero samples")
    return {key: value / samples for key, value in totals.items()} | {"samples": samples}


def log_validation_to_wandb(run, record: dict, video_paths=None) -> None:
    if run is None:
        return
    excluded = {"epoch", "epoch_fraction", "global_step", "optimizer_step", "trigger"}
    payload = {
        "optimizer_step": record["optimizer_step"],
        "validation/epoch": record["epoch_fraction"],
        **{
            f"validation/{key}": value
            for key, value in record.items()
            if key not in excluded and isinstance(value, (int, float))
        },
    }
    if video_paths:
        import wandb

        payload.update(
            {
                f"validation/sample_{index:02d}": wandb.Video(str(path), format="mp4")
                for index, path in enumerate(video_paths)
            }
        )
    run.log(payload, step=record["global_step"])


def main() -> None:
    implementation.load_config = load_config
    implementation.validate_model_geometry = validate_model_geometry
    implementation.LatentShardDataset = LatentShardDataset
    implementation.cache_collate = cache_collate
    implementation.CosmosContinuousTokenizer = RecordingCosmosTokenizer
    implementation.validate = validate
    implementation.log_validation_to_wandb = log_validation_to_wandb
    implementation.push_checkpoint_to_hub = push_checkpoint_to_hub
    implementation.start_wandb = start_wandb
    implementation.main()


if __name__ == "__main__":
    main()
