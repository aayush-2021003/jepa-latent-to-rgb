"""Train the V-JEPA 2.1 predicted-feature adapter through a frozen image decoder."""
from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.vjepa21_jepawms.adapter import VJEPA21ToJEPAWMSAdapter
from experiments.vjepa21_jepawms.common import (
    atomic_json_dump,
    atomic_torch_save,
    load_config,
    project_path,
    require_cuda,
    seed_everything,
)
from experiments.vjepa21_jepawms.data import TwoFrameCacheDataset, cache_collate
from experiments.vjepa21_jepawms.losses import (
    PerceptualLoss,
    feature_statistics_loss,
    image_metrics,
    latent_metrics,
)
from experiments.vjepa21_jepawms.media import save_comparison
from experiments.vjepa21_jepawms.models import FrozenJEPAWMSImageDecoder, validate_geometry
from experiments.vjepa21_jepawms.tracking import push_checkpoint
from experiments.jepa_cosmos.tracking import log_file_artifact, start_wandb


def build_adapter(config: dict) -> VJEPA21ToJEPAWMSAdapter:
    cfg = config["adapter"]
    return VJEPA21ToJEPAWMSAdapter(
        levels=cfg["levels"],
        level_dim=cfg["level_dim"],
        output_dim=cfg["output_dim"],
        residual_blocks=cfg["residual_blocks"],
        expansion=cfg["expansion"],
        dropout=cfg["dropout"],
        output_frames=cfg["output_frames"],
    )


def autocast_context(config: dict):
    enabled = config["training"]["mixed_precision"] == "bfloat16"
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def make_loader(config: dict, split: str, epoch: int = 0, limit: int | None = None):
    training = config["training"]
    dataset = TwoFrameCacheDataset(
        project_path(config["data"]["cache_root"]) / split,
        shuffle=split == "train",
        seed=config["experiment"]["seed"] + epoch,
        max_samples=limit,
    )
    return DataLoader(
        dataset,
        batch_size=training["batch_size"],
        num_workers=training["num_workers"] if split == "train" else 0,
        pin_memory=True,
        collate_fn=cache_collate,
        drop_last=False,
    )


@torch.no_grad()
def validate(
    config: dict,
    adapter,
    decoder,
    perceptual,
    device,
    step: int,
    output_dir: Path,
    run,
    limit: int,
    log_media: bool,
) -> dict[str, float]:
    adapter.eval()
    totals = {
        "predicted_l1": 0.0,
        "predicted_psnr": 0.0,
        "predicted_lpips": 0.0,
        "oracle_l1": 0.0,
        "oracle_psnr": 0.0,
        "oracle_lpips": 0.0,
        "last_frame_l1": 0.0,
        "last_frame_psnr": 0.0,
        "latent_l1": 0.0,
        "latent_cosine": 0.0,
    }
    for frame_number in (15, 16):
        for source in ("predicted", "oracle"):
            for metric in ("l1", "psnr", "lpips"):
                totals[f"frame{frame_number}_{source}_{metric}"] = 0.0
    samples = 0
    media_paths: list[Path] = []
    media_limit = config["training"]["media_samples"] if log_media else 0
    loader = make_loader(config, "val", limit=limit)
    for batch in loader:
        predicted_features = batch["predicted_features"].to(
            device, non_blocking=True
        )
        target_features = batch["target_features"].to(device, non_blocking=True)
        target = batch["target_rgb"].to(device, non_blocking=True).float() / 255.0
        context = batch["last_context_rgb"].to(device, non_blocking=True).float() / 255.0
        with autocast_context(config):
            prediction = decoder.decode_rgb01(adapter(predicted_features))
            oracle = decoder.decode_rgb01(adapter(target_features))
        pred_metrics = image_metrics(prediction, target)
        oracle_metrics = image_metrics(oracle, target)
        context_metrics = image_metrics(context, target[:, 1])
        latent = latent_metrics(predicted_features, target_features)
        batch_size = target.shape[0]
        flat_target = target.flatten(0, 1)
        pred_lpips = perceptual(prediction.flatten(0, 1), flat_target).item()
        oracle_lpips = perceptual(oracle.flatten(0, 1), flat_target).item()
        totals["predicted_l1"] += pred_metrics["l1"].sum().item()
        totals["predicted_psnr"] += pred_metrics["psnr"].sum().item()
        totals["predicted_lpips"] += pred_lpips * batch_size
        totals["oracle_l1"] += oracle_metrics["l1"].sum().item()
        totals["oracle_psnr"] += oracle_metrics["psnr"].sum().item()
        totals["oracle_lpips"] += oracle_lpips * batch_size
        totals["last_frame_l1"] += context_metrics["l1"].sum().item()
        totals["last_frame_psnr"] += context_metrics["psnr"].sum().item()
        totals["latent_l1"] += latent["l1"].sum().item()
        totals["latent_cosine"] += latent["cosine"].sum().item()
        for frame_index, frame_number in enumerate((15, 16)):
            frame_pred = image_metrics(prediction[:, frame_index], target[:, frame_index])
            frame_oracle = image_metrics(oracle[:, frame_index], target[:, frame_index])
            totals[f"frame{frame_number}_predicted_l1"] += frame_pred["l1"].sum().item()
            totals[f"frame{frame_number}_predicted_psnr"] += frame_pred["psnr"].sum().item()
            totals[f"frame{frame_number}_oracle_l1"] += frame_oracle["l1"].sum().item()
            totals[f"frame{frame_number}_oracle_psnr"] += frame_oracle["psnr"].sum().item()
            totals[f"frame{frame_number}_predicted_lpips"] += (
                perceptual(prediction[:, frame_index], target[:, frame_index]).item()
                * batch_size
            )
            totals[f"frame{frame_number}_oracle_lpips"] += (
                perceptual(oracle[:, frame_index], target[:, frame_index]).item()
                * batch_size
            )

        for index in range(batch_size):
            if len(media_paths) >= media_limit:
                break
            path = output_dir / "validation" / f"step_{step:07d}" / f"sample_{len(media_paths):02d}.png"
            save_comparison(
                path,
                batch["keys"][index],
                batch["last_context_rgb"][index],
                batch["target_rgb"][index],
                oracle[index],
                prediction[index],
            )
            media_paths.append(path)
        samples += batch_size

    if samples == 0:
        raise RuntimeError("Validation cache produced zero samples")
    metrics = {key: value / samples for key, value in totals.items()}
    metrics["samples"] = samples
    metrics["optimizer_step"] = step
    if run is not None:
        payload = {f"validation/{key}": value for key, value in metrics.items()}
        if media_paths:
            import wandb

            payload["validation/comparisons"] = [
                wandb.Image(str(path), caption=path.stem) for path in media_paths
            ]
        run.log(payload, step=step)
    adapter.train()
    return metrics


def save_checkpoint(
    path: Path,
    config: dict,
    adapter,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    best_metric: float,
    validation: dict,
) -> None:
    atomic_torch_save(
        {
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "optimizer_step": step,
            "best_metric": best_metric,
            "validation": validation,
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
        path,
    )


def write_history(history: list[dict], output_dir: Path) -> None:
    atomic_json_dump(history, output_dir / "history.json")
    if not history:
        return
    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        steps = [item["optimizer_step"] for item in history]
        axes[0].plot(steps, [item["frame16_predicted_l1"] for item in history], label="frame 16 predicted")
        axes[0].plot(steps, [item["frame16_oracle_l1"] for item in history], label="frame 16 oracle")
        axes[0].plot(steps, [item["last_frame_l1"] for item in history], label="last-frame copy")
        axes[0].set_title("Validation RGB L1")
        axes[1].plot(steps, [item["frame16_predicted_lpips"] for item in history], label="frame 16 predicted")
        axes[1].plot(steps, [item["frame16_oracle_lpips"] for item in history], label="frame 16 oracle")
        axes[1].set_title("Validation LPIPS")
        for axis in axes:
            axis.set_xlabel("optimizer step")
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "loss_curves.png", dpi=160)
        plt.close(figure)
    except Exception as error:
        print(f"Warning: could not render local curves: {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-hf", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_geometry(config)
    seed_everything(config["experiment"]["seed"])
    device = require_cuda()
    output_dir = project_path(args.output_dir or config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "adapter_latest.pt"
    best_path = output_dir / "adapter_best.pt"
    if args.fresh and (latest_path.exists() or best_path.exists()):
        raise RuntimeError(
            "--fresh will not overwrite an existing run. Pass --output-dir with a new directory."
        )

    adapter = build_adapter(config).to(device)
    decoder = FrozenJEPAWMSImageDecoder(config, device)
    perceptual = PerceptualLoss(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    train_samples = len(
        TwoFrameCacheDataset(
            project_path(config["data"]["cache_root"]) / "train",
            shuffle=False,
            seed=0,
        )
    )
    batches_per_epoch = math.ceil(train_samples / training["batch_size"])
    updates_per_epoch = math.ceil(
        batches_per_epoch / training["gradient_accumulation_steps"]
    )
    total_updates = updates_per_epoch * training["epochs"]
    min_ratio = training["min_learning_rate"] / training["learning_rate"]

    def lr_multiplier(step: int) -> float:
        progress = min(step / max(total_updates, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    start_epoch = 0
    optimizer_step = 0
    best_metric = float("inf")
    history: list[dict] = []
    last_validation: dict = {}
    if training["resume"] and latest_path.exists() and not args.fresh:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        optimizer_step = int(checkpoint["optimizer_step"])
        best_metric = float(checkpoint["best_metric"])
        last_validation = checkpoint.get("validation", {})
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
        print(f"Resumed from epoch {start_epoch}, optimizer step {optimizer_step}")

    run = None
    if not args.no_wandb:
        run = start_wandb(
            config,
            "adapter-training",
            config["tracking"]["run_name"],
        )
        run.define_metric("optimizer_step")
        run.define_metric("train/*", step_metric="optimizer_step")
        run.define_metric("validation/*", step_metric="optimizer_step")

    loss_cfg = config["loss"]
    accumulation = training["gradient_accumulation_steps"]
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, training["epochs"]):
        adapter.train()
        loader = make_loader(config, "train", epoch=epoch)
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{training['epochs']}")
        for batch_index, batch in enumerate(progress):
            features = batch["predicted_features"].to(device, non_blocking=True)
            target = batch["target_rgb"].to(device, non_blocking=True).float() / 255.0
            with autocast_context(config):
                adapted = adapter(features)
                # Keep RGB L1 gradients outside the display range. LPIPS still
                # receives clamped images, matching its expected [-1, 1] input.
                prediction = decoder.decode_rgb01(adapted, clamp=False)
                rgb_l1 = (prediction - target).abs().mean()
                perceptual_value = perceptual(
                    prediction.clamp(0.0, 1.0).flatten(0, 1),
                    target.flatten(0, 1),
                )
                statistics = feature_statistics_loss(adapted)
                loss = (
                    loss_cfg["rgb_l1"] * rgb_l1
                    + loss_cfg["perceptual"] * perceptual_value
                    + loss_cfg["feature_statistics"] * statistics
                )
            (loss / accumulation).backward()
            last_batch = batch_index + 1 == len(loader)
            if (batch_index + 1) % accumulation != 0 and not last_batch:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), training["max_grad_norm"]
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", step=optimizer_step)

            if run is not None and optimizer_step % config["tracking"]["log_every_optimizer_steps"] == 0:
                run.log(
                    {
                        "optimizer_step": optimizer_step,
                        "train/loss": loss.item(),
                        "train/rgb_l1": rgb_l1.item(),
                        "train/lpips": perceptual_value.item(),
                        "train/feature_statistics": statistics.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/gradient_norm": float(grad_norm),
                        "train/epoch": epoch + 1,
                    },
                    step=optimizer_step,
                )

            if optimizer_step % training["validate_every_optimizer_steps"] == 0:
                log_media = (
                    config["tracking"]["log_validation_media"]
                    and optimizer_step % training["media_every_optimizer_steps"] == 0
                )
                last_validation = validate(
                    config,
                    adapter,
                    decoder,
                    perceptual,
                    device,
                    optimizer_step,
                    output_dir,
                    run,
                    training["validation_samples"],
                    log_media,
                )
                history.append(dict(last_validation))
                write_history(history, output_dir)
                selected = float(last_validation[training["selection_metric"]])
                if selected < best_metric:
                    best_metric = selected
                    save_checkpoint(
                        best_path,
                        config,
                        adapter,
                        optimizer,
                        scheduler,
                        epoch,
                        optimizer_step,
                        best_metric,
                        last_validation,
                    )
                    if not args.no_hf and config["huggingface"]["push_best"]:
                        push_checkpoint(config, best_path, last_validation, "best")

            if optimizer_step % training["checkpoint_every_optimizer_steps"] == 0:
                save_checkpoint(
                    latest_path,
                    config,
                    adapter,
                    optimizer,
                    scheduler,
                    epoch,
                    optimizer_step,
                    best_metric,
                    last_validation,
                )
                if not args.no_hf and config["huggingface"]["push_latest"]:
                    push_checkpoint(config, latest_path, last_validation, "latest")

        save_checkpoint(
            latest_path,
            config,
            adapter,
            optimizer,
            scheduler,
            epoch,
            optimizer_step,
            best_metric,
            last_validation,
        )

    final_validation = validate(
        config,
        adapter,
        decoder,
        perceptual,
        device,
        optimizer_step,
        output_dir,
        run,
        training["final_validation_samples"],
        config["tracking"]["log_validation_media"],
    )
    atomic_json_dump(final_validation, output_dir / "final_validation.json")
    if not args.no_hf:
        if config["huggingface"]["push_latest"]:
            push_checkpoint(config, latest_path, final_validation, "latest")
        if config["huggingface"]["push_best"] and best_path.exists():
            best_payload = torch.load(
                best_path, map_location="cpu", weights_only=False
            ).get("validation", final_validation)
            push_checkpoint(config, best_path, best_payload, "best")
    if run is not None:
        artifacts = [
            output_dir / "history.json",
            output_dir / "history.csv",
            output_dir / "loss_curves.png",
            output_dir / "final_validation.json",
        ]
        log_file_artifact(run, "vjepa21-two-frame-results", "results", artifacts)
        run.finish()
    print(json.dumps(final_validation, indent=2))


if __name__ == "__main__":
    main()
