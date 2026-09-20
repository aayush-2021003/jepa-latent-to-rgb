"""Train the JEPA-to-Cosmos adapter and publish run artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.jepa_cosmos.adapter import build_adapter
from experiments.jepa_cosmos.common import (
    atomic_json_dump,
    atomic_torch_save,
    load_config,
    project_path,
    require_cuda,
    seed_everything,
)
from experiments.jepa_cosmos.data import LatentShardDataset, cache_collate
from experiments.jepa_cosmos.losses import (
    PerceptualVideoLoss,
    decoded_video_losses,
    latent_alignment_loss,
)
from experiments.jepa_cosmos.media import write_comparison_video
from experiments.jepa_cosmos.models import CosmosContinuousTokenizer, validate_model_geometry
from experiments.jepa_cosmos.tracking import (
    log_file_artifact,
    log_video,
    push_checkpoint_to_hub,
    start_wandb,
)


def dtype_from_name(name: str) -> torch.dtype:
    choices = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in choices:
        raise ValueError(f"Unsupported mixed precision dtype: {name}")
    return choices[name]


def enable_overfit_mode(config: dict) -> dict:
    """Apply a deliberately tiny, isolated memorization-test configuration."""
    if "overfit" not in config:
        raise KeyError("Config is missing the required 'overfit' section")
    mode = config["overfit"]
    config["experiment"]["name"] += "-overfit"
    config["experiment"]["output_dir"] = str(
        Path(config["experiment"]["output_dir"]) / "overfit"
    )
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "validate_every_optimizer_steps",
        "num_workers",
    ):
        config["training"][key] = mode[key]
    # A copied overfit YAML may still contain the former 125-epoch value. Enforce
    # the requested 5,000-update schedule (16 samples / batch 2 = 8 updates/epoch).
    config["training"]["epochs"] = max(int(mode.get("epochs", 0)), 625)
    # Prevent an older copied YAML from silently selecting the original small
    # adapter. Changing these dimensions requires a fresh checkpoint directory.
    config["adapter"]["hidden_dim"] = max(
        int(config["adapter"]["hidden_dim"]), 512
    )
    config["adapter"]["residual_blocks"] = max(
        int(config["adapter"]["residual_blocks"]), 6
    )
    # Overfit mode is a visual pipeline diagnostic: always log previews at the
    # configured validation interval, even when an older copied YAML says false.
    config["tracking"]["log_validation_videos"] = True
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
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "best_metric": best_metric,
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
        path,
    )


@torch.no_grad()
def validate(adapter, loader, device, config) -> dict[str, float]:
    adapter.eval()
    loss_cfg = config["loss"]
    totals = {
        "oracle_loss": 0.0,
        "oracle_l1": 0.0,
        "oracle_cosine": 0.0,
        "predicted_loss": 0.0,
        "predicted_l1": 0.0,
        "predicted_cosine": 0.0,
    }
    samples = 0
    for batch in tqdm(loader, desc="Validating", unit="batch", leave=False):
        target_jepa = batch["jepa_target"].to(device, non_blocking=True)
        predicted_jepa = batch["jepa_predicted"].to(device, non_blocking=True)
        cosmos_target = batch["cosmos_target"].to(device, non_blocking=True)
        target_shape = tuple(cosmos_target.shape[-3:])
        precision_dtype = dtype_from_name(config["training"]["mixed_precision"])
        with torch.autocast("cuda", dtype=precision_dtype):
            oracle = adapter(target_jepa, target_shape)
            predicted = adapter(predicted_jepa, target_shape)
        oracle_loss, oracle_parts = latent_alignment_loss(
            oracle,
            cosmos_target,
            loss_cfg["latent_l1"],
            loss_cfg["latent_cosine"],
        )
        predicted_loss, predicted_parts = latent_alignment_loss(
            predicted,
            cosmos_target,
            loss_cfg["latent_l1"],
            loss_cfg["latent_cosine"],
        )
        batch_size = target_jepa.shape[0]
        samples += batch_size
        totals["oracle_loss"] += float(oracle_loss) * batch_size
        totals["oracle_l1"] += float(oracle_parts["latent_l1"]) * batch_size
        totals["oracle_cosine"] += float(oracle_parts["latent_cosine"]) * batch_size
        totals["predicted_loss"] += float(predicted_loss) * batch_size
        totals["predicted_l1"] += float(predicted_parts["latent_l1"]) * batch_size
        totals["predicted_cosine"] += float(predicted_parts["latent_cosine"]) * batch_size
    if samples == 0:
        raise RuntimeError("Validation cache yielded zero samples")
    return {key: value / samples for key, value in totals.items()} | {"samples": samples}


def validation_record(
    validation: dict[str, float],
    epoch: int,
    epoch_fraction: float,
    global_step: int,
    optimizer_step: int,
    trigger: str,
) -> dict:
    return {
        "epoch": epoch + 1,
        "epoch_fraction": epoch_fraction,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "trigger": trigger,
        **validation,
    }


def log_validation_to_wandb(
    run,
    record: dict,
    video_paths: list[Path] | None = None,
) -> None:
    if run is None:
        return
    payload = {
        "optimizer_step": record["optimizer_step"],
        "validation/epoch": record["epoch_fraction"],
        "validation/oracle_loss": record["oracle_loss"],
        "validation/oracle_l1": record["oracle_l1"],
        "validation/oracle_cosine": record["oracle_cosine"],
        "validation/predicted_loss": record["predicted_loss"],
        "validation/predicted_l1": record["predicted_l1"],
        "validation/predicted_cosine": record["predicted_cosine"],
        "validation/samples": record["samples"],
    }
    if video_paths:
        import wandb

        payload.update(
            {
                f"validation/sample_{index:02d}": wandb.Video(
                    str(path), fps=4, format="mp4"
                )
                for index, path in enumerate(video_paths)
            }
        )
    run.log(payload, step=record["global_step"])


@torch.no_grad()
def render_previews(adapter, decoder, config, output_dir: Path) -> list[Path]:
    preview_path = project_path(config["data"]["cache_root"]) / "val" / "preview.pt"
    if not preview_path.is_file():
        raise FileNotFoundError(f"Validation preview cache not found: {preview_path}")
    preview = torch.load(preview_path, map_location="cpu", weights_only=False)
    paths = []
    device = next(adapter.parameters()).device
    future_frames = config["data"]["num_frames"] - config["data"]["context_frames"]
    for index, sample in enumerate(preview):
        anchor = sample["cosmos_anchor"].unsqueeze(0).to(device)
        target = sample["cosmos_target"].unsqueeze(0).to(device)
        target_shape = tuple(target.shape[-3:])
        precision_dtype = dtype_from_name(config["training"]["mixed_precision"])
        with torch.autocast("cuda", dtype=precision_dtype):
            oracle_latent = adapter(sample["jepa_target"].unsqueeze(0).to(device), target_shape)
            predicted_latent = adapter(sample["jepa_predicted"].unsqueeze(0).to(device), target_shape)
        reconstruction = decoder.decode_anchored(anchor, target, future_frames)[0]
        oracle = decoder.decode_anchored(anchor, oracle_latent, future_frames)[0]
        predicted = decoder.decode_anchored(anchor, predicted_latent, future_frames)[0]
        path = output_dir / "validation_videos" / f"sample_{index:02d}.mp4"
        write_comparison_video(
            path,
            sample["future_rgb"].permute(1, 0, 2, 3),
            reconstruction,
            oracle,
            predicted,
            fps=4,
        )
        paths.append(path)
    return paths


def plot_history(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_oracle_loss"] for row in history], label="validation oracle")
    plt.plot(epochs, [row["val_predicted_loss"] for row in history], label="validation predicted")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Memorize a tiny shared train/validation subset before the full run.",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-hf-push", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.overfit:
        config = enable_overfit_mode(config)
    validate_model_geometry(config)
    seed_everything(config["experiment"]["seed"])
    device = require_cuda()
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    validation_interval = int(train_cfg["validate_every_optimizer_steps"])
    if validation_interval <= 0:
        raise ValueError("validate_every_optimizer_steps must be positive")
    output_dir = project_path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    overfit_samples = config["overfit"]["samples"] if args.overfit else None
    train_split = "val" if args.overfit else "train"
    train_dataset = LatentShardDataset(
        project_path(config["data"]["cache_root"]) / train_split,
        # Keep the bounded prefix fixed so overfit train and validation samples match.
        shuffle=not args.overfit,
        seed=config["experiment"]["seed"],
        max_samples=overfit_samples,
    )
    val_dataset = LatentShardDataset(
        project_path(config["data"]["cache_root"]) / "val",
        shuffle=False,
        seed=config["experiment"]["seed"],
        max_samples=overfit_samples,
    )
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
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_dataset)
        / train_cfg["batch_size"]
        / train_cfg["gradient_accumulation_steps"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, optimizer_steps_per_epoch * train_cfg["epochs"]),
        eta_min=train_cfg["min_learning_rate"],
    )
    precision_dtype = dtype_from_name(train_cfg["mixed_precision"])
    perceptual = None
    if loss_cfg["perceptual"] > 0:
        perceptual = PerceptualVideoLoss().to(device)
    decoded_enabled = any(
        loss_cfg[key] > 0 for key in ("rgb_l1", "perceptual", "temporal")
    )
    if decoded_enabled and train_cfg["decoded_loss_every_steps"] <= 0:
        raise ValueError("Decoded loss weights are non-zero but decoded_loss_every_steps is not positive")
    decoder = None
    if decoded_enabled or config["tracking"]["log_validation_videos"]:
        decoder = CosmosContinuousTokenizer(config, device, load_encoder=False, load_decoder=True)

    latest_path = output_dir / "adapter_latest.pt"
    best_path = output_dir / "adapter_best.pt"
    start_epoch = 0
    global_step = 0
    optimizer_step = 0
    best_metric = float("inf")
    if train_cfg["resume"] and latest_path.is_file():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        optimizer_step = state.get(
            "optimizer_step",
            global_step // train_cfg["gradient_accumulation_steps"],
        )
        best_metric = state["best_metric"]

    run_name = config["experiment"]["name"] if args.overfit else None
    run = None if args.no_wandb else start_wandb(config, "adapter-training", run_name)
    if run is not None:
        run.define_metric("optimizer_step")
        run.define_metric("validation/*", step_metric="optimizer_step")
    history_path = output_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    validation_history_path = output_dir / "validation_history.json"
    validation_history = (
        json.loads(validation_history_path.read_text())
        if validation_history_path.exists()
        else []
    )
    optimizer.zero_grad(set_to_none=True)
    future_frames = config["data"]["num_frames"] - config["data"]["context_frames"]

    for epoch in range(start_epoch, train_cfg["epochs"]):
        adapter.train()
        epoch_loss = 0.0
        samples = 0
        best_updated_this_epoch = False
        best_metrics_this_epoch = None
        last_validation_optimizer_step = -1
        validation = None
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}", unit="batch")
        for batch_index, batch in enumerate(progress):
            jepa_target = batch["jepa_target"].to(device, non_blocking=True)
            cosmos_anchor = batch["cosmos_anchor"].to(device, non_blocking=True)
            cosmos_target = batch["cosmos_target"].to(device, non_blocking=True)
            target_shape = tuple(cosmos_target.shape[-3:])
            with torch.autocast("cuda", dtype=precision_dtype):
                prediction = adapter(jepa_target, target_shape)
                loss, parts = latent_alignment_loss(
                    prediction,
                    cosmos_target,
                    loss_cfg["latent_l1"],
                    loss_cfg["latent_cosine"],
                )
                should_decode = (
                    decoded_enabled
                    and global_step % train_cfg["decoded_loss_every_steps"] == 0
                )
                if should_decode:
                    assert decoder is not None
                    predicted_video = decoder.decode_anchored_with_input_grad(
                        cosmos_anchor, prediction, future_frames
                    )
                    with torch.no_grad():
                        target_video = decoder.decode_anchored(
                            cosmos_anchor, cosmos_target, future_frames
                        )
                    decoded_loss, decoded_parts = decoded_video_losses(
                        predicted_video,
                        target_video,
                        loss_cfg["rgb_l1"],
                        loss_cfg["temporal"],
                    )
                    loss = loss + decoded_loss
                    parts.update(decoded_parts)
                    if perceptual is not None:
                        perceptual_value = perceptual(predicted_video, target_video)
                        loss = loss + loss_cfg["perceptual"] * perceptual_value
                        parts["perceptual"] = perceptual_value
                scaled_loss = loss / train_cfg["gradient_accumulation_steps"]
            scaled_loss.backward()
            batch_size = jepa_target.shape[0]
            should_step = (
                (batch_index + 1) % train_cfg["gradient_accumulation_steps"] == 0
                or samples + batch_size >= len(train_dataset)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), train_cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
            samples += batch_size
            epoch_loss += float(loss.detach()) * batch_size
            global_step += 1
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            if run is not None and global_step % config["tracking"]["log_every_steps"] == 0:
                metrics = {
                    "train/loss": float(loss.detach()),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch + 1,
                    "train/optimizer_step": optimizer_step,
                }
                metrics.update({f"train/{key}": float(value.detach()) for key, value in parts.items()})
                run.log(metrics, step=global_step)

            if should_step and optimizer_step % validation_interval == 0:
                validation = validate(adapter, val_loader, device, config)
                record = validation_record(
                    validation,
                    epoch,
                    epoch + samples / len(train_dataset),
                    global_step,
                    optimizer_step,
                    "optimizer_step",
                )
                validation_history.append(record)
                atomic_json_dump(validation_history, validation_history_path)
                interval_videos = None
                if (
                    args.overfit
                    and decoder is not None
                    and config["tracking"]["log_validation_videos"]
                ):
                    interval_videos = render_previews(
                        adapter, decoder, config, output_dir
                    )
                log_validation_to_wandb(run, record, interval_videos)
                last_validation_optimizer_step = optimizer_step
                if validation["oracle_loss"] < best_metric:
                    best_metric = validation["oracle_loss"]
                    best_updated_this_epoch = True
                    best_metrics_this_epoch = {**record, "best_metric": best_metric}
                    save_checkpoint(
                        best_path,
                        adapter,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        optimizer_step,
                        best_metric,
                        config,
                    )
                adapter.train()

        train_loss = epoch_loss / samples
        if last_validation_optimizer_step != optimizer_step:
            validation = validate(adapter, val_loader, device, config)
            record = validation_record(
                validation,
                epoch,
                epoch + 1.0,
                global_step,
                optimizer_step,
                "epoch_end",
            )
            validation_history.append(record)
            atomic_json_dump(validation_history, validation_history_path)
            log_validation_to_wandb(run, record)
            if validation["oracle_loss"] < best_metric:
                best_metric = validation["oracle_loss"]
                best_updated_this_epoch = True
                best_metrics_this_epoch = {**record, "best_metric": best_metric}
                save_checkpoint(
                    best_path,
                    adapter,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    optimizer_step,
                    best_metric,
                    config,
                )
        assert validation is not None
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_oracle_loss": validation["oracle_loss"],
            "val_predicted_loss": validation["predicted_loss"],
            "val_oracle_l1": validation["oracle_l1"],
            "val_predicted_l1": validation["predicted_l1"],
        }
        history.append(epoch_row)
        atomic_json_dump(history, history_path)
        save_checkpoint(
            latest_path,
            adapter,
            optimizer,
            scheduler,
            epoch,
            global_step,
            optimizer_step,
            best_metric,
            config,
        )
        metrics = {
            "epoch": epoch + 1,
            "optimizer_step": optimizer_step,
            "train_loss": train_loss,
            **validation,
            "best_metric": best_metric,
        }
        if run is not None:
            run.log({f"epoch/{key}": value for key, value in metrics.items()}, step=global_step)
        hf_push_enabled = (
            not args.no_hf_push
            and (not args.overfit or config["overfit"]["push_to_huggingface"])
        )
        if hf_push_enabled:
            if config["huggingface"]["push_latest"]:
                push_checkpoint_to_hub(config, latest_path, metrics, "latest")
            if best_updated_this_epoch and config["huggingface"]["push_best"]:
                assert best_metrics_this_epoch is not None
                url = push_checkpoint_to_hub(
                    config, best_path, best_metrics_this_epoch, "best"
                )
                if run is not None:
                    run.summary["huggingface_best_checkpoint"] = url

        if (
            best_updated_this_epoch
            and decoder is not None
            and config["tracking"]["log_validation_videos"]
            and not args.overfit
        ):
            best_state = torch.load(best_path, map_location="cpu", weights_only=False)
            adapter.load_state_dict(best_state["adapter"])
            videos = render_previews(adapter, decoder, config, output_dir)
            if run is not None:
                for index, path in enumerate(videos):
                    log_video(run, f"validation/sample_{index:02d}", path, fps=4)
            latest_state = torch.load(latest_path, map_location="cpu", weights_only=False)
            adapter.load_state_dict(latest_state["adapter"])

    curve_path = output_dir / "loss_curves.png"
    plot_history(history, curve_path)
    summary = {
        "mode": "overfit" if args.overfit else "full",
        "best_validation_oracle_loss": best_metric,
        "epochs_completed": len(history),
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "validate_every_optimizer_steps": validation_interval,
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
            "jepa-cosmos-training-results",
            "results",
            [history_path, validation_history_path, summary_path, curve_path],
        )
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
