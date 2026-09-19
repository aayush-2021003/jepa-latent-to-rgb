"""Phase 1: train an oracle full-FactorJEPA-grid -> Cosmos VAE latent decoder.

Official architecture references:
  https://github.com/facebookresearch/vjepa2
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py

The FactorJEPA encoder and Cosmos VAE are frozen. The decoder is trained only on
unmasked, full-video encoder features; predictor outputs never enter this script.
Use ``--experiment-mode overfit_one`` with three identical one-clip manifests for
the fixed-sample memorization diagnostic (VAE ceiling, initial, training, final).

USAGE
  python -u src/oracle_jepa_decoder_train.py \
    --mode poc \
    --experiment-mode generalization \
    --model-config configs/model/vjepa2_1_vitg.yaml \
    --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
    --decode-config configs/jepa_pixel_decode.yaml \
    --factorjepa-repo anonymousML123/factorjepa-outputs \
    --factorjepa-repo-type dataset \
    --factorjepa-filename outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt \
    --local-data data/full_local \
    --train-manifest data/jepa_decode_splits/poc/train.json \
    --validation-manifest data/jepa_decode_splits/poc/validation.json \
    --test-manifest data/jepa_decode_splits/poc/test.json \
    --output-dir outputs/jepa_pixel_decode/oracle_poc \
    --wandb-group factorjepa-one-sample \
    --cache-policy 2
"""
import argparse
import json
import math
import time
import traceback
from pathlib import Path

import torch

from utils.cache_policy import add_cache_policy_arg, resolve_cache_policy_interactive, wipe_output_dir
from utils.config import load_merged_config
from utils.jepa_pixel_decode import (
    LPIPSComputer,
    append_jsonl,
    atomic_json,
    atomic_torch_save,
    audit_source_disjoint,
    build_oracle_decoder,
    compute_oracle_training_loss,
    decode_cosmos_latents,
    encode_cosmos_latents,
    export_comparison_video,
    extract_oracle_features,
    iter_video_batches,
    load_cosmos_vae_only,
    load_factorjepa_checkpoint,
    per_clip_latent_metrics,
    per_clip_pixel_metrics,
    read_clip_manifest,
    read_jsonl_records,
    require_cuda,
    resolve_hf_checkpoint,
    sha256_file,
    summarize_records,
    write_csv,
)
from utils.progress import make_pbar
from utils.training import load_config
from utils.wandb_utils import (
    add_wandb_args,
    finish_wandb,
    init_wandb,
    log_artifact,
    log_metrics,
    log_video,
)


def make_scheduler(optimizer, total_steps: int, warmup_fraction: float):
    if total_steps <= 0:
        raise ValueError("total optimizer steps must be positive")
    if not 0.0 <= warmup_fraction <= 0.1:
        raise ValueError(f"warmup_fraction must be in [0, 0.1], got {warmup_fraction}")
    warmup_steps = round(total_steps * warmup_fraction)

    def multiplier(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), warmup_steps


def checkpoint_payload(*, decoder, optimizer, scheduler, epoch: int, global_step: int,
                       best_metric: float, stale_epochs: int, validation_pending: bool,
                       model_cfg: dict, data_cfg: dict,
                       decode_cfg: dict, latent_shape: tuple[int, int, int, int],
                       source_checkpoint: str, source_checkpoint_sha256: str,
                       load_report: dict, split_audit: dict) -> dict:
    return {
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "stale_epochs": stale_epochs,
        "validation_pending": validation_pending,
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
        "decode_cfg": decode_cfg,
        "latent_shape": list(latent_shape),
        "grid_shape": list(decoder.grid_shape),
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_size_bytes": Path(source_checkpoint).stat().st_size,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "factorjepa_load_report": load_report,
        "split_audit": split_audit,
        "experiment_contract": "oracle_full_encoder_features_to_cosmos_vae_latent",
    }


def flatten_summary_metrics(summary: dict, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/{name}": float(interval["mean"])
        for name, interval in summary["metrics"].items()
    }


@torch.no_grad()
def evaluate_overfit_one(*, epoch: int, clip_key: str, decoder, oracle_features,
                         target_latent, raw_batch, vae_video, initial_video,
                         cosmos, cfg: dict, decode_cfg: dict, output_dir: Path,
                         lpips_computer: LPIPSComputer, wandb_run) -> tuple[dict, float]:
    decoder.eval()
    predicted_latent = decoder(oracle_features)
    predicted_video = decode_cosmos_latents(
        cosmos, predicted_latent, int(cfg["data"]["num_frames"]), requires_grad=False)
    metric_groups = {
        "oracle_vs_target": per_clip_latent_metrics(predicted_latent, target_latent),
        "vae_vs_input": per_clip_pixel_metrics(
            vae_video, raw_batch, decode_cfg["metrics"], lpips_computer),
        "oracle_vs_input": per_clip_pixel_metrics(
            predicted_video, raw_batch, decode_cfg["metrics"], lpips_computer),
        "oracle_vs_vae": per_clip_pixel_metrics(
            predicted_video, vae_video, decode_cfg["metrics"], lpips_computer),
    }
    record = {"clip_key": clip_key, "epoch": epoch}
    for prefix, values in metric_groups.items():
        record.update({
            f"{prefix}_{name}": float(tensor[0].detach())
            for name, tensor in values.items()
        })
    records_path = output_dir / decode_cfg["artifacts"]["overfit_metrics_records"]
    append_jsonl(records_path, record)
    metric_names = sorted(key for key in record if key not in {"clip_key", "epoch"})
    summary = summarize_records([record], metric_names, decode_cfg["bootstrap"])
    summary.update({
        "clip_key": clip_key,
        "epoch": epoch,
        "phase": "one_sample_memorization_diagnostic",
        "generalization_claim_valid": False,
    })
    video_path = (
        output_dir / decode_cfg["artifacts"]["overfit_gallery_dir"] /
        decode_cfg["artifacts"]["overfit_gallery_pattern"].format(epoch=epoch)
    )
    export_comparison_video(
        [raw_batch[0], vae_video[0], initial_video[0], predicted_video[0]],
        ["input", "Cosmos VAE", "initial decoder", f"decoder epoch {epoch}"],
        video_path,
        int(decode_cfg["video"]["fps"]),
        decode_cfg["video"]["labels"],
    )
    log_metrics(
        wandb_run,
        flatten_summary_metrics(summary, "overfit/eval"),
        step=epoch,
        commit=False,
    )
    log_video(
        wandb_run,
        "overfit/reconstruction",
        str(video_path),
        int(decode_cfg["video"]["fps"]),
        step=epoch,
        commit=True,
    )
    decoder.train()
    return summary, float(record["oracle_vs_target_latent_mae"])


def overfit_checkpoint_payload(*, decoder, optimizer, scheduler, epoch: int,
                               best_latent_mae: float, clip_key: str,
                               model_cfg: dict, data_cfg: dict, decode_cfg: dict,
                               latent_shape: tuple[int, int, int, int],
                               source_checkpoint: str, source_checkpoint_sha256: str,
                               load_report: dict) -> dict:
    return {
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": epoch,
        "best_metric": best_latent_mae,
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
        "decode_cfg": decode_cfg,
        "latent_shape": list(latent_shape),
        "grid_shape": list(decoder.grid_shape),
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_size_bytes": Path(source_checkpoint).stat().st_size,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "factorjepa_load_report": load_report,
        "clip_key": clip_key,
        "experiment_contract": "one_sample_oracle_memorization_diagnostic",
        "generalization_claim_valid": False,
    }


def run_overfit_one(*, args, cfg: dict, decode_cfg: dict, device: torch.device,
                    student, cosmos, lpips_computer: LPIPSComputer,
                    output_dir: Path, source_checkpoint: str,
                    source_checkpoint_sha256: str, load_report: dict):
    train_keys = read_clip_manifest(args.train_manifest)
    validation_keys = read_clip_manifest(args.validation_manifest)
    test_keys = read_clip_manifest(args.test_manifest)
    if len(train_keys) != 1 or train_keys != validation_keys or train_keys != test_keys:
        raise RuntimeError(
            "overfit_one requires train, validation, and test manifests to contain "
            "the same single clip key"
        )
    clip_key = train_keys[0]
    diagnostic_audit = {
        "experiment_mode": "overfit_one",
        "clip_key": clip_key,
        "unique_clips": 1,
        "source_disjoint": False,
        "generalization_claim_valid": False,
    }
    atomic_json(output_dir / decode_cfg["artifacts"]["split_audit"], diagnostic_audit)
    wandb_run = init_wandb(
        module_name="oracle_jepa_decoder_overfit_one",
        mode=args.mode.upper(),
        config={
            "experiment_mode": args.experiment_mode,
            "clip_key": clip_key,
            "model_config": args.model_config,
            "train_config": args.train_config,
            "decode_config": args.decode_config,
            "factorjepa_filename": args.factorjepa_filename,
            "overfit_one": decode_cfg["overfit_one"],
        },
        enabled=not args.no_wandb,
        group=args.wandb_group,
    )
    try:
        batches = list(iter_video_batches(
            local_data=args.local_data,
            clip_keys=train_keys,
            num_frames=int(cfg["data"]["num_frames"]),
            crop_size=int(cfg["model"]["crop_size"]),
            batch_size=1,
            num_readers=int(decode_cfg["data_loading"]["tar_readers"]),
            decode_workers=int(decode_cfg["data_loading"]["decode_workers"]),
            queue_timeout_seconds=int(decode_cfg["data_loading"]["queue_timeout_seconds"]),
            reader_join_timeout_seconds=int(
                decode_cfg["data_loading"]["reader_join_timeout_seconds"]),
            training=False,
            augmentation_cfg=cfg["augmentation"],
        ))
        if len(batches) != 1 or batches[0][1] != [clip_key]:
            raise RuntimeError("one-sample loader did not return exactly the requested clip")
        raw_batch = batches[0][0].to(device)
        with torch.no_grad():
            oracle_features = extract_oracle_features(
                raw_batch, student, cfg["model"], decode_cfg["factorjepa"]).detach()
            target_latent = encode_cosmos_latents(cosmos, raw_batch).detach()
            vae_video = decode_cosmos_latents(
                cosmos, target_latent, int(cfg["data"]["num_frames"]), requires_grad=False)

        latent_shape = tuple(target_latent.shape[1:])
        latest_path = output_dir / decode_cfg["artifacts"]["overfit_latest_checkpoint"]
        best_path = output_dir / decode_cfg["artifacts"]["overfit_best_checkpoint"]
        final_path = output_dir / decode_cfg["artifacts"]["overfit_final_checkpoint"]
        resume = torch.load(
            latest_path, map_location="cpu", weights_only=False) if latest_path.exists() else None
        decoder = build_oracle_decoder(
            cfg["model"], cfg["data"], decode_cfg, latent_shape).to(device)
        with torch.no_grad():
            initial_latent = decoder(oracle_features)
            initial_video = decode_cosmos_latents(
                cosmos, initial_latent, int(cfg["data"]["num_frames"]), requires_grad=False)
        overfit_cfg = decode_cfg["overfit_one"]
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=float(overfit_cfg["learning_rate"]),
            weight_decay=float(overfit_cfg["weight_decay"]),
        )
        max_epochs = int(overfit_cfg["max_epochs"])
        scheduler, warmup_epochs = make_scheduler(
            optimizer, max_epochs, float(overfit_cfg["warmup_fraction"]))
        start_epoch = 0
        best_latent_mae = float("inf")
        if resume is not None:
            if resume["experiment_contract"] != "one_sample_oracle_memorization_diagnostic":
                raise RuntimeError("latest checkpoint is not an overfit-one diagnostic checkpoint")
            if resume["clip_key"] != clip_key:
                raise RuntimeError("resume checkpoint belongs to a different clip")
            if resume["model_cfg"] != cfg["model"] or resume["data_cfg"] != cfg["data"]:
                raise RuntimeError("resume checkpoint FactorJEPA configuration differs")
            if resume["decode_cfg"] != decode_cfg:
                raise RuntimeError("resume checkpoint decode configuration differs")
            if resume["source_checkpoint_sha256"] != source_checkpoint_sha256:
                raise RuntimeError("resume checkpoint FactorJEPA SHA-256 differs")
            decoder.load_state_dict(resume["decoder_state_dict"], strict=True)
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scheduler.load_state_dict(resume["scheduler_state_dict"])
            start_epoch = int(resume["epoch"])
            best_latent_mae = float(resume["best_metric"])

        if start_epoch == 0:
            initial_summary, initial_mae = evaluate_overfit_one(
                epoch=0,
                clip_key=clip_key,
                decoder=decoder,
                oracle_features=oracle_features,
                target_latent=target_latent,
                raw_batch=raw_batch,
                vae_video=vae_video,
                initial_video=initial_video,
                cosmos=cosmos,
                cfg=cfg,
                decode_cfg=decode_cfg,
                output_dir=output_dir,
                lpips_computer=lpips_computer,
                wandb_run=wandb_run,
            )
            best_latent_mae = initial_mae
            atomic_json(
                output_dir / decode_cfg["artifacts"]["overfit_metrics_summary"], initial_summary)
            initial_payload = overfit_checkpoint_payload(
                decoder=decoder,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=0,
                best_latent_mae=best_latent_mae,
                clip_key=clip_key,
                model_cfg=cfg["model"],
                data_cfg=cfg["data"],
                decode_cfg=decode_cfg,
                latent_shape=latent_shape,
                source_checkpoint=source_checkpoint,
                source_checkpoint_sha256=source_checkpoint_sha256,
                load_report=load_report,
            )
            atomic_torch_save(latest_path, initial_payload)
            atomic_torch_save(best_path, initial_payload)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = output_dir / f"{decode_cfg['artifacts']['overfit_log_prefix']}_{timestamp}.jsonl"
        loss_decode_cfg = dict(decode_cfg)
        loss_decode_cfg["loss"] = dict(decode_cfg["loss"])
        loss_decode_cfg["loss"]["decoded_loss_every_steps"] = int(
            overfit_cfg["decoded_loss_every_epochs"])
        evaluation_every = int(overfit_cfg["evaluation_every_epochs"])
        checkpoint_every = int(overfit_cfg["checkpoint_every_epochs"])
        latent_mae_target = float(overfit_cfg["latent_mae_target"])
        progress = make_pbar(
            total=max_epochs - start_epoch,
            desc=f"overfit one clip ({clip_key})",
            unit="epoch",
        )
        last_payload = None
        for epoch in range(start_epoch + 1, max_epochs + 1):
            decoder.train()
            optimizer.zero_grad(set_to_none=True)
            predicted_latent = decoder(oracle_features)
            loss, loss_values = compute_oracle_training_loss(
                predicted_latent=predicted_latent,
                target_latent=target_latent,
                raw_batch=raw_batch,
                cosmos=cosmos,
                decode_cfg=loss_decode_cfg,
                lpips_computer=lpips_computer,
                global_step=epoch,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), float(overfit_cfg["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            record = {
                "epoch": epoch,
                "global_step": epoch,
                "batch_size": 1,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": float(gradient_norm.detach()),
                **loss_values,
            }
            append_jsonl(log_path, record)
            should_evaluate = epoch % evaluation_every == 0 or epoch == max_epochs
            log_metrics(
                wandb_run,
                {f"overfit/train/{name}": value for name, value in record.items()
                 if name not in {"epoch", "global_step"}},
                step=epoch,
                commit=not should_evaluate,
            )
            current_mae = None
            if should_evaluate:
                summary, current_mae = evaluate_overfit_one(
                    epoch=epoch,
                    clip_key=clip_key,
                    decoder=decoder,
                    oracle_features=oracle_features,
                    target_latent=target_latent,
                    raw_batch=raw_batch,
                    vae_video=vae_video,
                    initial_video=initial_video,
                    cosmos=cosmos,
                    cfg=cfg,
                    decode_cfg=decode_cfg,
                    output_dir=output_dir,
                    lpips_computer=lpips_computer,
                    wandb_run=wandb_run,
                )
                improved = current_mae < best_latent_mae
                if improved:
                    best_latent_mae = current_mae
                summary["target_reached"] = current_mae <= latent_mae_target
                atomic_json(
                    output_dir / decode_cfg["artifacts"]["overfit_metrics_summary"], summary)
            last_payload = overfit_checkpoint_payload(
                decoder=decoder,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_latent_mae=best_latent_mae,
                clip_key=clip_key,
                model_cfg=cfg["model"],
                data_cfg=cfg["data"],
                decode_cfg=decode_cfg,
                latent_shape=latent_shape,
                source_checkpoint=source_checkpoint,
                source_checkpoint_sha256=source_checkpoint_sha256,
                load_report=load_report,
            )
            if epoch % checkpoint_every == 0 or should_evaluate:
                atomic_torch_save(latest_path, last_payload)
            if should_evaluate and current_mae == best_latent_mae:
                atomic_torch_save(best_path, last_payload)
            progress.set_postfix_str(
                f"loss={loss_values['total']:.5f} best_mae={best_latent_mae:.5f}")
            progress.update(1)
            if current_mae is not None and current_mae <= latent_mae_target:
                break
        progress.close()
        if last_payload is None:
            last_payload = overfit_checkpoint_payload(
                decoder=decoder,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=start_epoch,
                best_latent_mae=best_latent_mae,
                clip_key=clip_key,
                model_cfg=cfg["model"],
                data_cfg=cfg["data"],
                decode_cfg=decode_cfg,
                latent_shape=latent_shape,
                source_checkpoint=source_checkpoint,
                source_checkpoint_sha256=source_checkpoint_sha256,
                load_report=load_report,
            )
        atomic_torch_save(latest_path, last_payload)
        atomic_torch_save(final_path, last_payload)
        if not best_path.exists():
            atomic_torch_save(best_path, last_payload)
        metrics_records_path = (
            output_dir / decode_cfg["artifacts"]["overfit_metrics_records"])
        metrics_summary_path = (
            output_dir / decode_cfg["artifacts"]["overfit_metrics_summary"])
        log_artifact(wandb_run, "overfit-one-best-checkpoint", str(best_path), "model")
        log_artifact(wandb_run, "overfit-one-final-checkpoint", str(final_path), "model")
        log_artifact(wandb_run, "overfit-one-training-jsonl", str(log_path), "training")
        log_artifact(
            wandb_run, "overfit-one-metrics-jsonl", str(metrics_records_path), "evaluation")
        log_artifact(
            wandb_run, "overfit-one-summary", str(metrics_summary_path), "evaluation")
        print(
            f"Finished one-sample diagnostic: clip={clip_key} "
            f"epoch={last_payload['epoch']} best_latent_mae={best_latent_mae:.6f}"
        )
    finally:
        finish_wandb(wandb_run)


@torch.no_grad()
def validate_oracle(*, epoch: int, decoder, student, cosmos, cfg: dict, decode_cfg: dict,
                    local_data: str, validation_keys: list[str], output_dir: Path,
                    lpips_computer: LPIPSComputer):
    decoder.eval()
    epoch_name = decode_cfg["artifacts"]["oracle_epoch_dir_pattern"].format(epoch=epoch)
    epoch_dir = output_dir / decode_cfg["artifacts"]["oracle_validation_dir"] / epoch_name
    records_path = epoch_dir / decode_cfg["artifacts"]["oracle_validation_records"]
    records = read_jsonl_records(records_path, ("clip_key",))
    completed = {record["clip_key"] for record in records}
    unexpected = completed - set(validation_keys)
    if unexpected:
        raise RuntimeError(f"validation records contain {len(unexpected)} unexpected clip keys")
    remaining_keys = [key for key in validation_keys if key not in completed]
    gallery_index = len(records)
    batch_iterator = iter_video_batches(
            local_data=local_data,
            clip_keys=remaining_keys,
            num_frames=int(cfg["data"]["num_frames"]),
            crop_size=int(cfg["model"]["crop_size"]),
            batch_size=int(decode_cfg["validation"]["batch_size"]),
            num_readers=int(decode_cfg["data_loading"]["tar_readers"]),
            decode_workers=int(decode_cfg["data_loading"]["decode_workers"]),
            queue_timeout_seconds=int(decode_cfg["data_loading"]["queue_timeout_seconds"]),
            reader_join_timeout_seconds=int(
                decode_cfg["data_loading"]["reader_join_timeout_seconds"]),
            training=False,
            augmentation_cfg=cfg["augmentation"]) if remaining_keys else ()
    progress = make_pbar(
        total=len(remaining_keys), desc=f"oracle validation epoch {epoch}", unit="clip")
    for raw_batch, batch_keys in batch_iterator:
        raw_batch = raw_batch.to(next(decoder.parameters()).device)
        oracle_features = extract_oracle_features(
            raw_batch, student, cfg["model"], decode_cfg["factorjepa"])
        target_latent = encode_cosmos_latents(cosmos, raw_batch)
        oracle_latent = decoder(oracle_features)
        vae_video = decode_cosmos_latents(
            cosmos, target_latent, int(cfg["data"]["num_frames"]), requires_grad=False)
        oracle_video = decode_cosmos_latents(
            cosmos, oracle_latent, int(cfg["data"]["num_frames"]), requires_grad=False)

        metric_groups = {
            "oracle_vs_target": per_clip_latent_metrics(oracle_latent, target_latent),
            "vae_vs_input": per_clip_pixel_metrics(
                vae_video, raw_batch, decode_cfg["metrics"], lpips_computer),
            "oracle_vs_input": per_clip_pixel_metrics(
                oracle_video, raw_batch, decode_cfg["metrics"], lpips_computer),
            "oracle_vs_vae": per_clip_pixel_metrics(
                oracle_video, vae_video, decode_cfg["metrics"], lpips_computer),
        }
        for batch_index, clip_key in enumerate(batch_keys):
            record = {"clip_key": clip_key, "epoch": epoch}
            for prefix, values in metric_groups.items():
                record.update({
                    f"{prefix}_{name}": float(tensor[batch_index].detach())
                    for name, tensor in values.items()
                })
            if gallery_index < int(decode_cfg["validation"]["qualitative_clips"]):
                export_comparison_video(
                    [raw_batch[batch_index], vae_video[batch_index], oracle_video[batch_index]],
                    ["input", "Cosmos VAE", "oracle JEPA"],
                    epoch_dir / decode_cfg["artifacts"]["oracle_gallery_dir"] /
                    decode_cfg["artifacts"]["oracle_gallery_pattern"].format(index=gallery_index),
                    int(decode_cfg["video"]["fps"]),
                    decode_cfg["video"]["labels"],
                )
            append_jsonl(records_path, record)
            records.append(record)
            gallery_index += 1
        progress.update(len(batch_keys))
    progress.close()

    if {record["clip_key"] for record in records} != set(validation_keys):
        raise RuntimeError("oracle validation did not produce every expected clip record")
    metric_names = sorted({key for record in records for key in record if key not in {"clip_key", "epoch"}})
    summary = summarize_records(records, metric_names, decode_cfg["bootstrap"])
    summary.update({"epoch": epoch, "phase": "oracle_validation"})
    write_csv(epoch_dir / decode_cfg["artifacts"]["oracle_validation_records_csv"], records)
    atomic_json(epoch_dir / decode_cfg["artifacts"]["oracle_validation_summary"], summary)
    decoder.train()
    return summary


def main():
    parser = argparse.ArgumentParser("Train the isolated oracle FactorJEPA pixel decoder")
    parser.add_argument("--mode", required=True, choices=["sanity", "poc", "full"])
    parser.add_argument(
        "--experiment-mode", required=True, choices=["generalization", "overfit_one"])
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--decode-config", required=True)
    parser.add_argument("--factorjepa-repo", required=True)
    parser.add_argument("--factorjepa-repo-type", required=True)
    parser.add_argument("--factorjepa-filename", required=True)
    parser.add_argument("--local-data", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-group", required=True)
    add_cache_policy_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    cfg = load_merged_config(args.model_config, args.train_config)
    decode_cfg = load_config(args.decode_config)
    device = require_cuda()
    torch.manual_seed(int(cfg["data"]["seed"]))
    torch.cuda.manual_seed_all(int(cfg["data"]["seed"]))

    train_keys = read_clip_manifest(args.train_manifest)
    validation_keys = read_clip_manifest(args.validation_manifest)
    test_keys = read_clip_manifest(args.test_manifest)
    split_audit = (
        audit_source_disjoint(train_keys, validation_keys, test_keys)
        if args.experiment_mode == "generalization"
        else {
            "experiment_mode": "overfit_one",
            "audit_deferred_to_overfit_contract": True,
        }
    )
    print(json.dumps(split_audit, indent=2))

    output_dir = Path(args.output_dir)
    wipe_output_dir(output_dir, args.cache_policy, label="oracle JEPA decoder output")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / decode_cfg["artifacts"]["split_audit"], split_audit)

    source_checkpoint = resolve_hf_checkpoint(
        args.factorjepa_repo, args.factorjepa_repo_type, args.factorjepa_filename)
    source_checkpoint_sha256 = sha256_file(
        source_checkpoint, int(decode_cfg["provenance"]["sha256_chunk_size_bytes"]))
    student, _, load_report = load_factorjepa_checkpoint(
        checkpoint_path=source_checkpoint,
        model_cfg=cfg["model"],
        data_cfg=cfg["data"],
        load_predictor=False,
        device=device,
        factor_cfg=decode_cfg["factorjepa"],
    )
    cosmos = load_cosmos_vae_only(decode_cfg["cosmos"], device)
    lpips_computer = LPIPSComputer(decode_cfg["metrics"], device)

    if args.experiment_mode == "overfit_one":
        run_overfit_one(
            args=args,
            cfg=cfg,
            decode_cfg=decode_cfg,
            device=device,
            student=student,
            cosmos=cosmos,
            lpips_computer=lpips_computer,
            output_dir=output_dir,
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha256=source_checkpoint_sha256,
            load_report=load_report,
        )
        return

    batch_size = int(decode_cfg["optimization"]["batch_size"])
    epochs = int(decode_cfg["optimization"]["max_epochs"][args.mode])
    steps_per_epoch = math.ceil(len(train_keys) / batch_size)
    total_steps = epochs * steps_per_epoch
    latest_path = output_dir / decode_cfg["artifacts"]["oracle_latest_checkpoint"]
    best_path = output_dir / decode_cfg["artifacts"]["oracle_best_checkpoint"]
    final_path = output_dir / decode_cfg["artifacts"]["oracle_final_checkpoint"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"{decode_cfg['artifacts']['oracle_log_prefix']}_{timestamp}.jsonl"
    resume = torch.load(latest_path, map_location="cpu", weights_only=False) if latest_path.exists() else None

    decoder = None
    optimizer = None
    scheduler = None
    latent_shape = None
    start_epoch = int(resume["epoch"]) if resume is not None else 0
    global_step = int(resume["global_step"]) if resume is not None else 0
    best_metric = float(resume["best_metric"]) if resume is not None else float("inf")
    stale_epochs = int(resume["stale_epochs"]) if resume is not None else 0
    last_payload = resume

    if resume is not None:
        if resume["model_cfg"] != cfg["model"] or resume["data_cfg"] != cfg["data"]:
            raise RuntimeError("resume checkpoint FactorJEPA model/data config differs from this run")
        if resume["decode_cfg"] != decode_cfg:
            raise RuntimeError("resume checkpoint decode config differs from this run")
        if int(resume["source_checkpoint_size_bytes"]) != Path(source_checkpoint).stat().st_size:
            raise RuntimeError("resume checkpoint was trained against a different FactorJEPA file size")
        if resume["source_checkpoint_sha256"] != source_checkpoint_sha256:
            raise RuntimeError("resume checkpoint was trained against a different FactorJEPA SHA-256")
        latent_shape = tuple(resume["latent_shape"])
        decoder = build_oracle_decoder(
            cfg["model"], cfg["data"], decode_cfg, latent_shape).to(device)
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=float(decode_cfg["optimization"]["learning_rate"]),
            weight_decay=float(decode_cfg["optimization"]["weight_decay"]),
        )
        scheduler, warmup_steps = make_scheduler(
            optimizer, total_steps, float(decode_cfg["optimization"]["warmup_fraction"]))
        decoder.load_state_dict(resume["decoder_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        print(
            f"Resumed oracle decoder at epoch={start_epoch} step={global_step}; "
            f"parameters={sum(p.numel() for p in decoder.parameters()):,} "
            f"total_steps={total_steps} warmup_steps={warmup_steps}"
        )

        if bool(resume["validation_pending"]):
            print(f"Completing pending validation for epoch={start_epoch} from the exact saved model state")
            validation_summary = validate_oracle(
                epoch=start_epoch,
                decoder=decoder,
                student=student,
                cosmos=cosmos,
                cfg=cfg,
                decode_cfg=decode_cfg,
                local_data=args.local_data,
                validation_keys=validation_keys,
                output_dir=output_dir,
                lpips_computer=lpips_computer,
            )
            validation_metric = decode_cfg["optimization"]["validation_metric"]
            metric_key = f"oracle_vs_target_{validation_metric}"
            current_metric = float(validation_summary["metrics"][metric_key]["mean"])
            improved = current_metric < best_metric - float(
                decode_cfg["optimization"]["early_stopping_min_delta"])
            if improved:
                best_metric = current_metric
                stale_epochs = 0
            else:
                stale_epochs += 1
            last_payload = checkpoint_payload(
                decoder=decoder,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=start_epoch,
                global_step=global_step,
                best_metric=best_metric,
                stale_epochs=stale_epochs,
                validation_pending=False,
                model_cfg=cfg["model"],
                data_cfg=cfg["data"],
                decode_cfg=decode_cfg,
                latent_shape=latent_shape,
                source_checkpoint=source_checkpoint,
                source_checkpoint_sha256=source_checkpoint_sha256,
                load_report=load_report,
                split_audit=split_audit,
            )
            atomic_torch_save(latest_path, last_payload)
            if improved:
                atomic_torch_save(best_path, last_payload)
            print(
                f"Epoch {start_epoch}: {metric_key}={current_metric:.6f} "
                f"best={best_metric:.6f} stale_epochs={stale_epochs}"
            )

        if stale_epochs >= int(decode_cfg["optimization"]["early_stopping_patience_epochs"]):
            atomic_torch_save(final_path, last_payload)
            print(f"Oracle training already early-stopped; saved final checkpoint to {final_path}")
            return

    for epoch in range(start_epoch, epochs):
        progress = make_pbar(total=steps_per_epoch, desc=f"oracle epoch {epoch + 1}/{epochs}", unit="step")
        for raw_batch, batch_keys in iter_video_batches(
                local_data=args.local_data,
                clip_keys=train_keys,
                num_frames=int(cfg["data"]["num_frames"]),
                crop_size=int(cfg["model"]["crop_size"]),
                batch_size=batch_size,
                num_readers=int(decode_cfg["data_loading"]["tar_readers"]),
                decode_workers=int(decode_cfg["data_loading"]["decode_workers"]),
                queue_timeout_seconds=int(decode_cfg["data_loading"]["queue_timeout_seconds"]),
                reader_join_timeout_seconds=int(
                    decode_cfg["data_loading"]["reader_join_timeout_seconds"]),
                training=True,
                augmentation_cfg=cfg["augmentation"]):
            raw_batch = raw_batch.to(device)
            with torch.no_grad():
                oracle_features = extract_oracle_features(
                    raw_batch, student, cfg["model"], decode_cfg["factorjepa"])
                target_latent = encode_cosmos_latents(cosmos, raw_batch)
            if decoder is None:
                latent_shape = tuple(target_latent.shape[1:])
                decoder = build_oracle_decoder(
                    cfg["model"], cfg["data"], decode_cfg, latent_shape).to(device)
                optimizer = torch.optim.AdamW(
                    decoder.parameters(),
                    lr=float(decode_cfg["optimization"]["learning_rate"]),
                    weight_decay=float(decode_cfg["optimization"]["weight_decay"]),
                )
                scheduler, warmup_steps = make_scheduler(
                    optimizer, total_steps, float(decode_cfg["optimization"]["warmup_fraction"]))
                print(
                    f"Oracle decoder parameters={sum(p.numel() for p in decoder.parameters()):,} "
                    f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps}"
                )

            decoder.train()
            optimizer.zero_grad(set_to_none=True)
            predicted_latent = decoder(oracle_features.detach())
            loss, loss_values = compute_oracle_training_loss(
                predicted_latent=predicted_latent,
                target_latent=target_latent,
                raw_batch=raw_batch,
                cosmos=cosmos,
                decode_cfg=decode_cfg,
                lpips_computer=lpips_computer,
                global_step=global_step,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), float(decode_cfg["optimization"]["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            global_step += 1
            record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "batch_size": len(batch_keys),
                "learning_rate": scheduler.get_last_lr()[0],
                **loss_values,
            }
            append_jsonl(log_path, record)
            progress.set_postfix_str(f"loss={loss_values['total']:.5f}")
            progress.update(1)
        progress.close()

        pending_payload = checkpoint_payload(
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
            best_metric=best_metric,
            stale_epochs=stale_epochs,
            validation_pending=True,
            model_cfg=cfg["model"],
            data_cfg=cfg["data"],
            decode_cfg=decode_cfg,
            latent_shape=latent_shape,
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha256=source_checkpoint_sha256,
            load_report=load_report,
            split_audit=split_audit,
        )
        atomic_torch_save(latest_path, pending_payload)

        validation_summary = validate_oracle(
            epoch=epoch + 1,
            decoder=decoder,
            student=student,
            cosmos=cosmos,
            cfg=cfg,
            decode_cfg=decode_cfg,
            local_data=args.local_data,
            validation_keys=validation_keys,
            output_dir=output_dir,
            lpips_computer=lpips_computer,
        )
        validation_metric = decode_cfg["optimization"]["validation_metric"]
        metric_key = f"oracle_vs_target_{validation_metric}"
        current_metric = float(validation_summary["metrics"][metric_key]["mean"])
        improved = current_metric < best_metric - float(
            decode_cfg["optimization"]["early_stopping_min_delta"])
        if improved:
            best_metric = current_metric
            stale_epochs = 0
        else:
            stale_epochs += 1
        last_payload = checkpoint_payload(
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
            best_metric=best_metric,
            stale_epochs=stale_epochs,
            validation_pending=False,
            model_cfg=cfg["model"],
            data_cfg=cfg["data"],
            decode_cfg=decode_cfg,
            latent_shape=latent_shape,
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha256=source_checkpoint_sha256,
            load_report=load_report,
            split_audit=split_audit,
        )
        atomic_torch_save(latest_path, last_payload)
        if improved:
            atomic_torch_save(best_path, last_payload)
        print(
            f"Epoch {epoch + 1}: {metric_key}={current_metric:.6f} "
            f"best={best_metric:.6f} stale_epochs={stale_epochs}"
        )
        if stale_epochs >= int(decode_cfg["optimization"]["early_stopping_patience_epochs"]):
            print(f"Early stopping after {stale_epochs} non-improving epochs")
            break

    if decoder is None or last_payload is None:
        raise RuntimeError("oracle training produced zero optimizer steps")
    atomic_torch_save(final_path, last_payload)
    print(f"Saved oracle decoder: best={best_path} final={final_path}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"FATAL (oracle-jepa-decoder): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
