"""Train a predicted-tubelet adapter against one frame's Cosmos-CI latent."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.jepa_cosmos.losses import latent_alignment_loss
from experiments.vjepa21_cosmos_single_frame.adapter import build_adapter
from experiments.vjepa21_cosmos_single_frame.common import (
    atomic_json_dump,
    atomic_torch_save,
    load_config,
    project_path,
    require_cuda,
    seed_everything,
)
from experiments.vjepa21_cosmos_single_frame.data import LatentShardDataset, cache_collate
from experiments.vjepa21_cosmos_single_frame.media import save_comparison
from experiments.vjepa21_cosmos_single_frame.models import (
    CosmosContinuousImageTokenizer,
    validate_model_geometry,
)
from experiments.vjepa21_cosmos_single_frame.tracking import (
    log_file_artifact,
    push_checkpoint_to_hub,
    start_wandb,
)


def dtype_from_name(name: str) -> torch.dtype:
    choices = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in choices:
        raise ValueError(f"Unsupported mixed precision dtype: {name}")
    return choices[name]


def enable_overfit_mode(config: dict) -> dict:
    if "overfit" not in config:
        raise KeyError("Config is missing the overfit section")
    mode = config["overfit"]
    mode["samples"] = 1
    config["experiment"]["name"] += "-overfit"
    config["experiment"]["output_dir"] = str(
        Path(config["experiment"]["output_dir"]) / "overfit"
    )
    config["training"].update(
        {
            "epochs": int(mode.get("epochs", 3000)),
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": float(mode.get("learning_rate", 1.0e-3)),
            "scheduler": "none",
            "validate_every_optimizer_steps": int(
                mode.get("validate_every_optimizer_steps", 50)
            ),
            "num_workers": 0,
        }
    )
    config["tracking"]["log_validation_media"] = True
    config["_overfit_mode"] = True
    return config


def save_checkpoint(
    path: Path,
    adapter,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    best_metric: float,
    config: dict,
) -> None:
    atomic_torch_save(
        {
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "best_metric": best_metric,
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
        path,
    )


def decoded_metrics(decoded: torch.Tensor, target_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = (decoded.float().clamp(-1.0, 1.0) + 1.0) / 2.0
    target = target_rgb.float() / 255.0
    delta = prediction - target
    l1 = delta.abs().mean(dim=(1, 2, 3))
    mse = delta.square().mean(dim=(1, 2, 3)).clamp_min(1.0e-10)
    return l1, -10.0 * torch.log10(mse)


@torch.no_grad()
def validate(adapter, loader, decoder, device, config) -> dict[str, float]:
    adapter.eval()
    totals = {
        "oracle_loss": 0.0,
        "oracle_l1": 0.0,
        "oracle_cosine": 0.0,
        "predicted_loss": 0.0,
        "predicted_l1": 0.0,
        "predicted_cosine": 0.0,
        "cosmos_reconstruction_rgb_l1": 0.0,
        "cosmos_reconstruction_rgb_psnr": 0.0,
        "oracle_rgb_l1": 0.0,
        "oracle_rgb_psnr": 0.0,
        "predicted_rgb_l1": 0.0,
        "predicted_rgb_psnr": 0.0,
    }
    samples = 0
    loss_cfg = config["loss"]
    precision = dtype_from_name(config["training"]["mixed_precision"])
    for batch in tqdm(loader, desc="Validating frame 15", unit="batch", leave=False):
        target_jepa = batch["jepa_target"].to(device, non_blocking=True)
        predicted_jepa = batch["jepa_predicted"].to(device, non_blocking=True)
        cosmos_target = batch["cosmos_target"].to(device, non_blocking=True)
        target_rgb = batch["target_rgb"].to(device, non_blocking=True)
        target_shape = tuple(cosmos_target.shape[-2:])
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
        reconstruction = decoder.decode(cosmos_target)
        oracle_rgb = decoder.decode(oracle_latent)
        predicted_rgb = decoder.decode(predicted_latent)
        reconstruction_l1, reconstruction_psnr = decoded_metrics(reconstruction, target_rgb)
        oracle_l1, oracle_psnr = decoded_metrics(oracle_rgb, target_rgb)
        predicted_l1, predicted_psnr = decoded_metrics(predicted_rgb, target_rgb)
        batch_size = target_jepa.shape[0]
        samples += batch_size
        values = {
            "oracle_loss": oracle_loss,
            "oracle_l1": oracle_parts["latent_l1"],
            "oracle_cosine": oracle_parts["latent_cosine"],
            "predicted_loss": predicted_loss,
            "predicted_l1": predicted_parts["latent_l1"],
            "predicted_cosine": predicted_parts["latent_cosine"],
        }
        for key, value in values.items():
            totals[key] += float(value) * batch_size
        totals["cosmos_reconstruction_rgb_l1"] += reconstruction_l1.sum().item()
        totals["cosmos_reconstruction_rgb_psnr"] += reconstruction_psnr.sum().item()
        totals["oracle_rgb_l1"] += oracle_l1.sum().item()
        totals["oracle_rgb_psnr"] += oracle_psnr.sum().item()
        totals["predicted_rgb_l1"] += predicted_l1.sum().item()
        totals["predicted_rgb_psnr"] += predicted_psnr.sum().item()
    if not samples:
        raise RuntimeError("Validation cache yielded zero samples")
    return {key: value / samples for key, value in totals.items()} | {"samples": samples}


@torch.no_grad()
def render_previews(adapter, decoder, config, output_dir: Path) -> list[Path]:
    preview_path = project_path(config["data"]["cache_root"]) / "val" / "preview.pt"
    if not preview_path.is_file():
        raise FileNotFoundError(f"Validation preview cache not found: {preview_path}")
    previews = torch.load(preview_path, map_location="cpu", weights_only=False)
    if config.get("_overfit_mode"):
        previews = previews[:1]
    device = next(adapter.parameters()).device
    precision = dtype_from_name(config["training"]["mixed_precision"])
    paths = []
    for index, sample in enumerate(previews):
        target = sample["cosmos_target"].unsqueeze(0).to(device)
        target_shape = tuple(target.shape[-2:])
        with torch.autocast("cuda", dtype=precision):
            oracle_latent = adapter(sample["jepa_target"].unsqueeze(0).to(device), target_shape)
            predicted_latent = adapter(
                sample["jepa_predicted"].unsqueeze(0).to(device), target_shape
            )
        reconstruction = decoder.decode(target)[0]
        oracle = decoder.decode(oracle_latent)[0]
        prediction = decoder.decode(predicted_latent)[0]
        path = output_dir / "validation_images" / f"sample_{index:02d}.png"
        save_comparison(
            path,
            sample["key"],
            sample["target_rgb"],
            reconstruction,
            oracle,
            prediction,
        )
        paths.append(path)
    return paths


def log_validation(run, record: dict, images: list[Path] | None = None) -> None:
    if run is None:
        return
    payload = {
        "optimizer_step": record["optimizer_step"],
        "validation/epoch": record["epoch_fraction"],
        **{
            f"validation/{key}": value
            for key, value in record.items()
            if key not in {"epoch", "epoch_fraction", "global_step", "optimizer_step", "trigger"}
            and isinstance(value, (int, float))
        },
    }
    if images:
        import wandb

        payload.update(
            {
                f"validation/sample_{index:02d}": wandb.Image(str(path))
                for index, path in enumerate(images)
            }
        )
    run.log(payload, step=record["global_step"])


def plot_history(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train predicted")
    plt.plot(epochs, [row["val_predicted_loss"] for row in history], label="validation predicted")
    plt.plot(epochs, [row["val_oracle_loss"] for row in history], label="validation oracle")
    plt.xlabel("Epoch")
    plt.ylabel("Latent alignment loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-hf-push", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.overfit:
        config = enable_overfit_mode(config)
    validate_model_geometry(config)
    if config["training"].get("input_latent") != "predicted":
        raise ValueError("This experiment requires training.input_latent=predicted")
    if config["training"].get("selection_metric") != "predicted_loss":
        raise ValueError("This experiment selects checkpoints by predicted_loss")
    if any(float(config["loss"].get(key, 0.0)) for key in ("rgb_l1", "perceptual")):
        raise ValueError("Training is latent-only; RGB metrics are validation-only")

    seed_everything(config["experiment"]["seed"])
    device = require_cuda()
    train_cfg = config["training"]
    validation_interval = int(train_cfg["validate_every_optimizer_steps"])
    if validation_interval <= 0:
        raise ValueError("validate_every_optimizer_steps must be positive")
    output_dir = project_path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_limit = int(config["overfit"]["samples"]) if args.overfit else None
    train_split = "val" if args.overfit else "train"
    train_dataset = LatentShardDataset(
        project_path(config["data"]["cache_root"]) / train_split,
        shuffle=not args.overfit,
        seed=config["experiment"]["seed"],
        max_samples=sample_limit,
    )
    val_dataset = LatentShardDataset(
        project_path(config["data"]["cache_root"]) / "val",
        shuffle=False,
        seed=config["experiment"]["seed"],
        max_samples=sample_limit,
    )
    for dataset in (train_dataset, val_dataset):
        if dataset.manifest.get("metadata", {}).get("contains_jepa_predicted") is not True:
            raise RuntimeError("Cache lacks predicted V-JEPA features")
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=cache_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        collate_fn=cache_collate,
    )
    adapter = build_adapter(config).to(device)
    decoder = CosmosContinuousImageTokenizer(config, device, False, True)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_dataset)
        / int(train_cfg["batch_size"])
        / int(train_cfg["gradient_accumulation_steps"])
    )
    scheduler_name = train_cfg.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, optimizer_steps_per_epoch * int(train_cfg["epochs"])),
            eta_min=float(train_cfg["min_learning_rate"]),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError("training.scheduler must be 'cosine' or 'none'")
    precision = dtype_from_name(train_cfg["mixed_precision"])
    latest_path = output_dir / "adapter_latest.pt"
    best_path = output_dir / "adapter_best.pt"
    start_epoch = global_step = optimizer_step = 0
    best_metric = float("inf")
    if train_cfg["resume"] and latest_path.is_file():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        optimizer_step = int(state["optimizer_step"])
        best_metric = float(state["best_metric"])

    run_name = config["tracking"]["run_name"] + ("-overfit" if args.overfit else "")
    run = None if args.no_wandb else start_wandb(
        config, "adapter-training", run_name
    )
    if run is not None:
        run.define_metric("optimizer_step")
        run.define_metric("validation/*", step_metric="optimizer_step")
    history_path = output_dir / "history.json"
    validation_path = output_dir / "validation_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    validation_history = json.loads(validation_path.read_text()) if validation_path.exists() else []
    optimizer.zero_grad(set_to_none=True)
    loss_cfg = config["loss"]
    completed_epochs = start_epoch

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        adapter.train()
        epoch_loss = 0.0
        samples = 0
        validation = None
        last_validation_step = -1
        best_updated = False
        best_record = None
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}", unit="batch")
        for batch_index, batch in enumerate(progress):
            jepa = batch["jepa_predicted"].to(device, non_blocking=True)
            target = batch["cosmos_target"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=precision):
                prediction = adapter(jepa, tuple(target.shape[-2:]))
                loss, parts = latent_alignment_loss(
                    prediction,
                    target,
                    float(loss_cfg["latent_l1"]),
                    float(loss_cfg["latent_cosine"]),
                )
                scaled_loss = loss / int(train_cfg["gradient_accumulation_steps"])
            scaled_loss.backward()
            batch_size = jepa.shape[0]
            should_step = (
                (batch_index + 1) % int(train_cfg["gradient_accumulation_steps"]) == 0
                or samples + batch_size >= len(train_dataset)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(train_cfg["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                optimizer_step += 1
            samples += batch_size
            epoch_loss += float(loss.detach()) * batch_size
            global_step += 1
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            if run is not None and global_step % int(config["tracking"]["log_every_steps"]) == 0:
                run.log(
                    {
                        "train/loss": float(loss.detach()),
                        "train/latent_l1": float(parts["latent_l1"]),
                        "train/latent_cosine": float(parts["latent_cosine"]),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch + 1,
                        "train/optimizer_step": optimizer_step,
                    },
                    step=global_step,
                )
            if should_step and optimizer_step % validation_interval == 0:
                validation = validate(adapter, val_loader, decoder, device, config)
                record = {
                    "epoch": epoch + 1,
                    "epoch_fraction": epoch + samples / len(train_dataset),
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "trigger": "optimizer_step",
                    **validation,
                }
                validation_history.append(record)
                atomic_json_dump(validation_history, validation_path)
                images = render_previews(adapter, decoder, config, output_dir) if config["tracking"]["log_validation_media"] else None
                log_validation(run, record, images)
                last_validation_step = optimizer_step
                if validation["predicted_loss"] < best_metric:
                    best_metric = validation["predicted_loss"]
                    best_updated = True
                    best_record = {**record, "best_metric": best_metric}
                    save_checkpoint(best_path, adapter, optimizer, scheduler, epoch, global_step, optimizer_step, best_metric, config)
                adapter.train()

        train_loss = epoch_loss / samples
        completed_epochs = epoch + 1
        final_epoch = completed_epochs == int(train_cfg["epochs"])
        validate_at_epoch_end = not args.overfit or final_epoch
        if validate_at_epoch_end and last_validation_step != optimizer_step:
            validation = validate(adapter, val_loader, decoder, device, config)
            record = {
                "epoch": epoch + 1,
                "epoch_fraction": epoch + 1.0,
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "trigger": "epoch_end",
                **validation,
            }
            validation_history.append(record)
            atomic_json_dump(validation_history, validation_path)
            final_images = (
                render_previews(adapter, decoder, config, output_dir)
                if args.overfit and config["tracking"]["log_validation_media"]
                else None
            )
            log_validation(run, record, final_images)
            if validation["predicted_loss"] < best_metric:
                best_metric = validation["predicted_loss"]
                best_updated = True
                best_record = {**record, "best_metric": best_metric}
                save_checkpoint(best_path, adapter, optimizer, scheduler, epoch, global_step, optimizer_step, best_metric, config)
        # In one-sample overfit mode, one epoch equals one optimizer step. Do not
        # let the generic epoch-end path turn a 50-step interval into validation
        # after every step. Checkpoint/history rows are emitted at validation
        # events (and the final step) only.
        if validation is None:
            continue
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_oracle_loss": validation["oracle_loss"],
                "val_predicted_loss": validation["predicted_loss"],
                "val_predicted_rgb_l1": validation["predicted_rgb_l1"],
                "val_predicted_rgb_psnr": validation["predicted_rgb_psnr"],
            }
        )
        atomic_json_dump(history, history_path)
        save_checkpoint(latest_path, adapter, optimizer, scheduler, epoch, global_step, optimizer_step, best_metric, config)
        metrics = {
            "epoch": epoch + 1,
            "optimizer_step": optimizer_step,
            "train_loss": train_loss,
            **validation,
            "best_metric": best_metric,
        }
        if run is not None:
            run.log({f"epoch/{key}": value for key, value in metrics.items()}, step=global_step)
        hf_enabled = not args.no_hf_push and (not args.overfit or config["overfit"]["push_to_huggingface"])
        if hf_enabled:
            if config["huggingface"]["push_latest"]:
                push_checkpoint_to_hub(config, latest_path, metrics, "latest")
            if best_updated and config["huggingface"]["push_best"]:
                url = push_checkpoint_to_hub(config, best_path, best_record, "best")
                if run is not None:
                    run.summary["huggingface_best_checkpoint"] = url

    curve_path = output_dir / "loss_curves.png"
    plot_history(history, curve_path)
    summary = {
        "mode": "overfit" if args.overfit else "predicted-tubelet-single-frame",
        "context_frames": 14,
        "predicted_tubelet_frames": [15, 16],
        "supervised_frame": 15,
        "scheduler": scheduler_name,
        "best_validation_predicted_loss": best_metric,
        "epochs_completed": completed_epochs,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "huggingface_repo": config["huggingface"]["repo_id"],
    }
    summary_path = output_dir / "training_summary.json"
    atomic_json_dump(summary, summary_path)
    if run is not None:
        import wandb

        run.log({"training/loss_curves": wandb.Image(str(curve_path))})
        log_file_artifact(
            run,
            "vjepa21-cosmos-ci-single-frame-results",
            "results",
            [history_path, validation_path, summary_path, curve_path],
        )
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
