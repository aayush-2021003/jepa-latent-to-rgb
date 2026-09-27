"""Predicted-latent frame-15 adapter with selectable latent/image supervision."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.jepa_cosmos.losses import latent_alignment_loss
from experiments.vjepa21_cosmos_single_frame.adapter import build_adapter
from experiments.vjepa21_cosmos_single_frame.common import (
    atomic_json_dump, load_config, project_path, require_cuda, seed_everything,
)
from experiments.vjepa21_cosmos_single_frame.data import LatentShardDataset, cache_collate
from experiments.vjepa21_cosmos_single_frame.models import (
    CosmosContinuousImageTokenizer, validate_model_geometry,
)
from experiments.vjepa21_cosmos_single_frame.train_adapter import (
    dtype_from_name, log_validation, render_previews, save_checkpoint,
)
from experiments.vjepa21_cosmos_single_frame.tracking import (
    log_file_artifact, push_checkpoint_to_hub, start_wandb,
)


def make_perceptual_model(config: dict, device: torch.device, *, evaluation: bool = False):
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("Install lpips before using image-loss training") from error
    loss_cfg = config["loss"]
    training_backbone = loss_cfg.get("perceptual_backbone", "alex")
    backbone = (loss_cfg.get("evaluation_perceptual_backbone", training_backbone)
                if evaluation else training_backbone)
    model = lpips.LPIPS(net=backbone)
    return model.to(device).eval().requires_grad_(False)


def objective(prediction, target_latent, target_rgb, decoder, perceptual, config):
    """RGB MSE is on [0,1]; LPIPS is on [-1,1]."""
    loss_cfg = config["loss"]
    mode = loss_cfg.get("objective", "latent_l1_cosine_rgb_lpips")
    if mode == "latent_mse":
        latent_mse = F.mse_loss(prediction.float(), target_latent.float())
        return latent_mse, {"latent_mse": latent_mse}
    if mode not in ("latent_l1_cosine_rgb_lpips", "latent_mse_rgb_lpips"):
        raise ValueError(f"Unsupported loss objective: {mode}")
    if mode == "latent_mse_rgb_lpips":
        weights = {name: float(loss_cfg[name]) for name in ("latent_mse", "rgb_mse", "perceptual")}
        if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        latent_loss = F.mse_loss(prediction.float(), target_latent.float())
        parts = {"latent_mse": latent_loss}
    else:
        latent_loss, parts = latent_alignment_loss(
            prediction, target_latent,
            float(loss_cfg["latent_l1"]), float(loss_cfg["latent_cosine"]),
        )
    decoded = decoder.decode(prediction)
    if torch.is_grad_enabled() and prediction.requires_grad and not decoded.requires_grad:
        raise RuntimeError("Cosmos decoder detached the adapter; RGB/LPIPS cannot train it")
    decoded = decoded.float()
    target_01 = target_rgb.float() / 255.0
    rgb_mse = F.mse_loss((decoded + 1.0) / 2.0, target_01)
    with torch.autocast("cuda", enabled=False):
        lpips_loss = perceptual(decoded.clamp(-1, 1), target_01 * 2.0 - 1.0).mean()
    if mode == "latent_mse_rgb_lpips":
        weighted = {
            "weighted_latent_mse": weights["latent_mse"] * latent_loss,
            "weighted_rgb_mse": weights["rgb_mse"] * rgb_mse,
            "weighted_rgb_lpips": weights["perceptual"] * lpips_loss,
        }
        return sum(weighted.values()), {**parts, "rgb_mse": rgb_mse,
                                         "rgb_lpips": lpips_loss, **weighted}
    total = (latent_loss + float(loss_cfg["rgb_mse"]) * rgb_mse
             + float(loss_cfg["perceptual"]) * lpips_loss)
    return total, {**parts, "rgb_mse": rgb_mse, "rgb_lpips": lpips_loss}


def load_adapter_initialization(adapter, config: dict, checkpoint_path: Path) -> dict:
    """Warm-start weights only; never inherit the source optimizer or scheduler."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Warm-start adapter checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "adapter" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Warm-start checkpoint must contain adapter weights and its config")
    source = checkpoint["config"]
    if source.get("training", {}).get("input_latent") != "predicted":
        raise ValueError("Warm-start checkpoint was not trained on predicted JEPA latents")
    if source.get("loss", {}).get("objective") != "latent_mse":
        raise ValueError("Expected a latent-MSE checkpoint for this fine-tuning experiment")
    for section, keys in (
        ("data", ("cache_root", "train_samples", "val_samples", "test_samples",
                  "context_frames", "target_frame_offset", "crop_size")),
        ("vjepa21", ("model_id", "embed_dim", "patch_size", "tubelet_size")),
        ("cosmos", ("model_id", "latent_channels", "spatial_compression")),
        ("adapter", ("input_dim", "hidden_dim", "output_channels", "residual_blocks")),
    ):
        for key in keys:
            old = source.get(section, {}).get(key)
            new = config[section][key]
            if old != new:
                raise ValueError(f"Warm-start checkpoint mismatch at {section}.{key}: {old!r} != {new!r}")
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "source_optimizer_step": checkpoint.get("optimizer_step"),
        "source_best_metric": checkpoint.get("best_metric"),
    }


def image_metrics(decoded, target_rgb, perceptual):
    prediction = ((decoded.float().clamp(-1, 1) + 1) / 2)
    target = target_rgb.float() / 255.0
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    lpips_value = perceptual(prediction * 2 - 1, target * 2 - 1).flatten(1).mean(1)
    return mse, -10 * torch.log10(mse.clamp_min(1e-10)), lpips_value


@torch.no_grad()
def evaluate(adapter, loader, decoder, perceptual, device, precision, static_metrics=None):
    adapter.eval()
    totals = {key: 0.0 for key in (
        "predicted_latent_l1", "predicted_latent_cosine", "predicted_latent_mse",
        "predicted_rgb_mse", "predicted_rgb_psnr", "predicted_rgb_lpips",
        "persistence_rgb_mse", "persistence_rgb_psnr", "persistence_rgb_lpips",
        "cosmos_rgb_mse", "cosmos_rgb_psnr", "cosmos_rgb_lpips",
    )}
    count = 0
    for batch in tqdm(loader, desc="Evaluating frame 15", leave=False):
        jepa = batch["jepa_predicted"].to(device, non_blocking=True)
        target_latent = batch["cosmos_target"].to(device, non_blocking=True)
        target_rgb = batch["target_rgb"].to(device, non_blocking=True)
        previous_rgb = (batch["previous_rgb"].to(device, non_blocking=True)
                        if static_metrics is None else None)
        with torch.autocast("cuda", dtype=precision):
            predicted_latent = adapter(jepa, tuple(target_latent.shape[-2:]))
            predicted_rgb = decoder.decode(predicted_latent)
            cosmos_rgb = decoder.decode(target_latent) if static_metrics is None else None
        _, latent_parts = latent_alignment_loss(predicted_latent, target_latent, 1, 1)
        metrics = {
            "predicted_latent_l1": latent_parts["latent_l1"],
            "predicted_latent_cosine": latent_parts["latent_cosine"],
            "predicted_latent_mse": F.mse_loss(predicted_latent.float(), target_latent.float()),
        }
        images = [("predicted", predicted_rgb)]
        if static_metrics is None:
            images.extend((("cosmos", cosmos_rgb),
                           ("persistence", previous_rgb.float() / 127.5 - 1)))
        for prefix, rgb in images:
            mse, psnr, lpips_value = image_metrics(rgb, target_rgb, perceptual)
            metrics[f"{prefix}_rgb_mse"] = mse.sum()
            metrics[f"{prefix}_rgb_psnr"] = psnr.sum()
            metrics[f"{prefix}_rgb_lpips"] = lpips_value.sum()
        batch_size = jepa.shape[0]
        count += batch_size
        for key, value in metrics.items():
            totals[key] += float(value) * (batch_size if key.startswith("predicted_latent") else 1)
    if count == 0:
        raise RuntimeError("Evaluation cache is empty")
    if static_metrics is not None:
        for key, value in static_metrics.items():
            totals[key] = value * count
    return {key: value / count for key, value in totals.items()} | {"samples": count}


def make_loader(config, split, shuffle):
    dataset = LatentShardDataset(
        project_path(config["data"]["cache_root"]) / split,
        shuffle=shuffle, seed=int(config["experiment"]["seed"]),
    )
    metadata = dataset.manifest.get("metadata", {})
    if metadata.get("contains_jepa_predicted") is not True or metadata.get("contains_target_rgb") is not True:
        raise RuntimeError(f"{split} cache lacks predicted JEPA latents or frame-15 RGB")
    if split != "train" and metadata.get("contains_previous_rgb") is not True:
        raise RuntimeError(f"{split} cache lacks frame-14 persistence baseline")
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]), pin_memory=True,
        collate_fn=cache_collate,
    )
    return dataset, loader


def plot_curves(history, output, selection_metric):
    if not history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train total loss")
    selected_key = f"val_{selection_metric}"
    if selection_metric == "predicted_latent_mse" and all(selected_key in row for row in history):
        axes[0].plot(epochs, [row[selected_key] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Training objective")
    axes[0].legend()
    axes[1].plot([r["epoch"] for r in history], [r["val_predicted_rgb_lpips"] for r in history], label="Predicted")
    axes[1].plot([r["epoch"] for r in history], [r["val_persistence_rgb_lpips"] for r in history], label="Persistence")
    axes[1].set(xlabel="Epoch", ylabel="Validation LPIPS")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-hf-push", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_model_geometry(config)
    if config["loss"].get("objective", "latent_l1_cosine_rgb_lpips") not in (
        "latent_l1_cosine_rgb_lpips", "latent_mse", "latent_mse_rgb_lpips"
    ):
        raise ValueError("Unknown loss objective")
    selection_metric = config["training"]["selection_metric"]
    if config["training"]["input_latent"] != "predicted" or selection_metric not in (
        "predicted_rgb_lpips", "predicted_latent_mse"
    ):
        raise ValueError("Expected predicted JEPA input and a supported checkpoint metric")
    if int(config["data"].get("test_samples", 0)) < 1:
        raise ValueError("A disjoint test split is required")
    seed_everything(config["experiment"]["seed"])
    device = require_cuda()
    output = project_path(config["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    train_cfg = config["training"]
    _, train_loader = make_loader(config, "train", True)
    _, val_loader = make_loader(config, "val", False)
    _, test_loader = make_loader(config, "test", False)
    adapter = build_adapter(config).to(device)
    decoder = CosmosContinuousImageTokenizer(config, device, False, True)
    perceptual = make_perceptual_model(config, device)
    training_backbone = config["loss"].get("perceptual_backbone", "alex")
    evaluation_backbone = config["loss"].get("evaluation_perceptual_backbone", training_backbone)
    evaluation_perceptual = (
        make_perceptual_model(config, device, evaluation=True)
        if evaluation_backbone != training_backbone else perceptual
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    steps_per_epoch = math.ceil(len(train_loader.dataset) / train_cfg["batch_size"] / train_cfg["gradient_accumulation_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps_per_epoch * train_cfg["epochs"]), eta_min=float(train_cfg["min_learning_rate"]))
    precision = dtype_from_name(train_cfg["mixed_precision"])
    best_path, latest_path = output / "adapter_best.pt", output / "adapter_latest.pt"
    start_epoch = global_step = optimizer_step = 0
    best_metric = float("inf")
    initialization = None
    if train_cfg["resume"] and latest_path.is_file() and not args.test_only:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        optimizer_step = int(checkpoint["optimizer_step"])
        best_metric = float(checkpoint["best_metric"])
    elif not args.test_only and train_cfg.get("init_adapter_checkpoint"):
        if best_path.is_file() or latest_path.is_file():
            raise RuntimeError(
                f"Refusing to overwrite an existing run in {output}; use resume or a new output_dir"
            )
        source_path = project_path(train_cfg["init_adapter_checkpoint"])
        if source_path.resolve() in (best_path.resolve(), latest_path.resolve()):
            raise ValueError("Warm-start checkpoint must be outside the new run's checkpoint paths")
        initialization = load_adapter_initialization(adapter, config, source_path)
        atomic_json_dump(initialization, output / "initialization.json")
    run = None if args.no_wandb else start_wandb(config, "adapter-image-training", config["tracking"]["run_name"])
    if run:
        run.define_metric("optimizer_step")
        run.define_metric("validation/*", step_metric="optimizer_step")
        run.summary["checkpoint_selection_metric"] = selection_metric
        run.summary["training_lpips_backbone"] = training_backbone
        run.summary["evaluation_lpips_backbone"] = evaluation_backbone
        run.summary["hf_model_repo"] = f"https://huggingface.co/{config['huggingface']['repo_id']}"
        if train_cfg.get("init_adapter_repo_id"):
            run.summary["source_adapter_repo"] = f"https://huggingface.co/{train_cfg['init_adapter_repo_id']}"
        if initialization:
            run.summary["initial_adapter_checkpoint"] = initialization["checkpoint"]
            run.summary["initial_adapter_optimizer_step"] = initialization["source_optimizer_step"]
    history_path, validation_path = output / "history.json", output / "validation_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    validation_history = json.loads(validation_path.read_text()) if validation_path.exists() else []
    interval = int(train_cfg["validate_every_optimizer_steps"])
    if interval < 1:
        raise ValueError("validate_every_optimizer_steps must be positive")

    validation_static = None

    def check_validation(epoch, fraction, trigger):
        nonlocal best_metric, validation_static
        metrics = evaluate(adapter, val_loader, decoder, evaluation_perceptual, device, precision,
                           static_metrics=validation_static)
        if validation_static is None:
            validation_static = {key: value for key, value in metrics.items()
                                 if key.startswith(("cosmos_", "persistence_"))}
        record = {"epoch": epoch, "epoch_fraction": fraction, "global_step": global_step,
                  "optimizer_step": optimizer_step, "trigger": trigger, **metrics}
        validation_history.append(record)
        atomic_json_dump(validation_history, validation_path)
        media = render_previews(adapter, decoder, config, output) if config["tracking"]["log_validation_media"] else None
        log_validation(run, record, media)
        improved = metrics[selection_metric] < best_metric
        if improved:
            best_metric = metrics[selection_metric]
            save_checkpoint(best_path, adapter, optimizer, scheduler, max(0, epoch - 1), global_step,
                            optimizer_step, best_metric, config)
            if run:
                run.summary["best_validation_metric"] = best_metric
                run.summary["best_optimizer_step"] = optimizer_step
            if not args.no_hf_push and config["huggingface"]["push_best"]:
                best_url = push_checkpoint_to_hub(config, best_path, record, "best")
                if run:
                    run.summary["hf_best_checkpoint"] = best_url
        adapter.train()
        return metrics

    if not args.test_only and start_epoch == 0 and optimizer_step == 0:
        check_validation(0, 0.0, "initial")

    for epoch in range(start_epoch, int(train_cfg["epochs"])) if not args.test_only else ():
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        samples, loss_total, last_validation, last_metrics = 0, 0.0, -1, None
        n_batches = math.ceil(len(train_loader.dataset) / train_cfg["batch_size"])
        accumulation = int(train_cfg["gradient_accumulation_steps"])
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")):
            jepa = batch["jepa_predicted"].to(device, non_blocking=True)
            target = batch["cosmos_target"].to(device, non_blocking=True)
            rgb = batch["target_rgb"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=precision):
                prediction = adapter(jepa, tuple(target.shape[-2:]))
                loss, parts = objective(prediction, target, rgb, decoder, perceptual, config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at optimizer step {optimizer_step}")
            group_start = (batch_index // accumulation) * accumulation
            divisor = min(accumulation, n_batches - group_start)
            (loss / divisor).backward()
            batch_size = jepa.shape[0]
            samples += batch_size
            loss_total += float(loss.detach()) * batch_size
            global_step += 1
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == n_batches:
                norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(train_cfg["max_grad_norm"]))
                if not torch.isfinite(norm):
                    raise RuntimeError(f"Non-finite adapter gradient at optimizer step {optimizer_step}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
                if optimizer_step % interval == 0:
                    last_metrics = check_validation(epoch + 1, epoch + samples / len(train_loader.dataset), "optimizer_step")
                    last_validation = optimizer_step
            if run and global_step % int(config["tracking"]["log_every_steps"]) == 0:
                run.log({"train/loss": float(loss.detach()), **{f"train/{key}": float(value.detach()) for key, value in parts.items()},
                         "train/lr": optimizer.param_groups[0]["lr"], "optimizer_step": optimizer_step}, step=global_step)
        if samples != len(train_loader.dataset):
            raise RuntimeError(f"Epoch yielded {samples} clips; expected {len(train_loader.dataset)}")
        metrics = (last_metrics if last_validation == optimizer_step else
                   check_validation(epoch + 1, epoch + 1.0, "epoch_end"))
        history.append({"epoch": epoch + 1, "train_loss": loss_total / samples,
                        **{f"val_{key}": value for key, value in metrics.items()}})
        atomic_json_dump(history, history_path)
        save_checkpoint(latest_path, adapter, optimizer, scheduler, epoch, global_step,
                        optimizer_step, best_metric, config)
        if config["huggingface"].get("push_every_epoch", False):
            epoch_path = output / f"adapter_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(epoch_path, adapter, optimizer, scheduler, epoch, global_step,
                            optimizer_step, best_metric, config)
            if not args.no_hf_push:
                epoch_record = {**history[-1], "optimizer_step": optimizer_step,
                                "selection_metric": selection_metric,
                                "best_validation_metric": best_metric}
                epoch_url = push_checkpoint_to_hub(
                    config, epoch_path, epoch_record, f"epoch_{epoch + 1:03d}"
                )
                if run:
                    run.summary[f"hf_epoch_{epoch + 1:03d}"] = epoch_url
        if not args.no_hf_push and config["huggingface"]["push_latest"]:
            latest_url = push_checkpoint_to_hub(config, latest_path, history[-1], "latest")
            if run:
                run.summary["hf_latest_checkpoint"] = latest_url
        if run:
            run.summary["last_completed_epoch"] = epoch + 1
            run.log({"epoch/train_loss": loss_total / samples,
                     **{f"epoch/val_{key}": value for key, value in metrics.items()},
                     f"epoch/best_val_{selection_metric}": best_metric,
                     "epoch/checkpoint_saved": epoch + 1,
                     "optimizer_step": optimizer_step}, step=global_step)

    if not best_path.is_file():
        raise FileNotFoundError(f"Best checkpoint unavailable for final test: {best_path}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(best["adapter"])
    test_metrics = evaluate(adapter, test_loader, decoder, evaluation_perceptual, device, precision)
    test_report = {"checkpoint": str(best_path), "selected_on": f"validation/{selection_metric}",
                   "best_validation_metric": float(best["best_metric"]), **test_metrics}
    report_path = output / "test_metrics.json"
    atomic_json_dump(test_report, report_path)
    test_media = render_previews(adapter, decoder, config, output, split="test") if config["tracking"]["log_validation_media"] else []
    curves = output / "loss_curves.png"
    plot_curves(history, curves, selection_metric)
    if run:
        import wandb
        payload = {f"test/{key}": value for key, value in test_metrics.items()}
        payload.update({f"test/sample_{index:02d}": wandb.Video(str(path), format="mp4") for index, path in enumerate(test_media) if path.suffix == ".mp4"})
        payload["training/loss_curves"] = wandb.Image(str(curves))
        run.log(payload, step=global_step)
        run.summary["test_predicted_rgb_lpips"] = test_metrics["predicted_rgb_lpips"]
        log_file_artifact(run, f"{config['experiment']['name']}-results", "results",
                          [history_path, validation_path, report_path, curves,
                           output / "initialization.json"])
        run.finish()
    print(json.dumps(test_report, indent=2))


if __name__ == "__main__":
    main()
