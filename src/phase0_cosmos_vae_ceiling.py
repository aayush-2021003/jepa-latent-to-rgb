"""Phase 0: measure the Cosmos VAE reconstruction ceiling on a pinned manifest.

Gold standard:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py

USAGE
  python -u src/phase0_cosmos_vae_ceiling.py \
    --model-config configs/model/vjepa2_1_vitg.yaml \
    --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
    --decode-config configs/jepa_pixel_decode.yaml \
    --local-data data/full_local \
    --manifest data/jepa_decode_splits/poc/validation.json \
    --output-dir outputs/jepa_pixel_decode/phase0_poc \
    --wandb-group factorjepa-one-sample \
    --cache-policy 2
"""
import argparse
import json
import traceback
from pathlib import Path

from utils.cache_policy import add_cache_policy_arg, resolve_cache_policy_interactive, wipe_output_dir
from utils.config import load_merged_config
from utils.jepa_pixel_decode import (
    LPIPSComputer,
    append_jsonl,
    atomic_json,
    decode_cosmos_latents,
    encode_cosmos_latents,
    export_comparison_video,
    iter_video_batches,
    load_cosmos_vae_only,
    per_clip_pixel_metrics,
    read_clip_manifest,
    read_jsonl_records,
    require_cuda,
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


def main():
    parser = argparse.ArgumentParser("Measure the Cosmos VAE video reconstruction ceiling")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--decode-config", required=True)
    parser.add_argument("--local-data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-group", required=True)
    add_cache_policy_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    cfg = load_merged_config(args.model_config, args.train_config)
    decode_cfg = load_config(args.decode_config)
    device = require_cuda()
    output_dir = Path(args.output_dir)
    wipe_output_dir(output_dir, args.cache_policy, label="Phase-0 Cosmos VAE ceiling output")
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(
        module_name="phase0_cosmos_vae_ceiling",
        mode="EVAL",
        config={
            "phase": "phase0",
            "manifest": args.manifest,
            "model_config": args.model_config,
            "train_config": args.train_config,
            "decode_config": args.decode_config,
        },
        enabled=not args.no_wandb,
        group=args.wandb_group,
    )

    clip_keys = read_clip_manifest(args.manifest)
    cosmos = load_cosmos_vae_only(decode_cfg["cosmos"], device)
    lpips_computer = LPIPSComputer(decode_cfg["metrics"], device)
    records_path = output_dir / decode_cfg["artifacts"]["phase0_records"]
    gallery_dir = output_dir / decode_cfg["artifacts"]["phase0_gallery_dir"]
    records = read_jsonl_records(records_path, ("clip_key",))
    completed = {record["clip_key"] for record in records}
    unexpected = completed - set(clip_keys)
    if unexpected:
        raise RuntimeError(f"Phase-0 records contain {len(unexpected)} unexpected clip keys")
    remaining_keys = [key for key in clip_keys if key not in completed]
    clip_index = len(records)
    existing_gallery_count = min(
        clip_index, int(decode_cfg["phase0"]["qualitative_clips"]))
    for existing_index in range(existing_gallery_count):
        existing_video_path = (
            gallery_dir / decode_cfg["artifacts"]["phase0_gallery_pattern"].format(
                index=existing_index)
        )
        log_video(
            wandb_run,
            "phase0/cosmos_vae_roundtrip",
            str(existing_video_path),
            int(decode_cfg["video"]["fps"]),
        )

    batch_iterator = iter_video_batches(
            local_data=args.local_data,
            clip_keys=remaining_keys,
            num_frames=int(cfg["data"]["num_frames"]),
            crop_size=int(cfg["model"]["crop_size"]),
            batch_size=int(decode_cfg["phase0"]["batch_size"]),
            num_readers=int(decode_cfg["data_loading"]["tar_readers"]),
            decode_workers=int(decode_cfg["data_loading"]["decode_workers"]),
            queue_timeout_seconds=int(decode_cfg["data_loading"]["queue_timeout_seconds"]),
            reader_join_timeout_seconds=int(
                decode_cfg["data_loading"]["reader_join_timeout_seconds"]),
            training=False,
            augmentation_cfg=cfg["augmentation"]) if remaining_keys else ()
    progress = make_pbar(total=len(remaining_keys), desc="Cosmos VAE ceiling", unit="clip")
    for raw_batch, batch_keys in batch_iterator:
        raw_batch = raw_batch.to(device)
        target_latent = encode_cosmos_latents(cosmos, raw_batch)
        reconstructed = decode_cosmos_latents(
            cosmos, target_latent, int(cfg["data"]["num_frames"]), requires_grad=False)
        batch_metrics = per_clip_pixel_metrics(
            reconstructed, raw_batch, decode_cfg["metrics"], lpips_computer)
        for batch_index, clip_key in enumerate(batch_keys):
            record = {
                "clip_key": clip_key,
                **{name: float(values[batch_index].detach()) for name, values in batch_metrics.items()},
            }
            log_metrics(
                wandb_run,
                {f"phase0/per_clip/{name}": value for name, value in record.items()
                 if name != "clip_key"},
                step=clip_index,
                commit=clip_index >= int(decode_cfg["phase0"]["qualitative_clips"]),
            )
            if clip_index < int(decode_cfg["phase0"]["qualitative_clips"]):
                video_path = (
                    gallery_dir / decode_cfg["artifacts"]["phase0_gallery_pattern"].format(
                        index=clip_index)
                )
                export_comparison_video(
                    [raw_batch[batch_index], reconstructed[batch_index]],
                    ["input", "Cosmos VAE roundtrip"],
                    video_path,
                    int(decode_cfg["video"]["fps"]),
                    decode_cfg["video"]["labels"],
                )
                log_video(
                    wandb_run,
                    "phase0/cosmos_vae_roundtrip",
                    str(video_path),
                    int(decode_cfg["video"]["fps"]),
                    step=clip_index,
                    commit=True,
                )
            append_jsonl(records_path, record)
            records.append(record)
            clip_index += 1
        progress.update(len(batch_keys))
    progress.close()

    metric_names = ["pixel_mse", "pixel_mae", "psnr_db", "temporal_difference_l1", "lpips"]
    if {record["clip_key"] for record in records} != set(clip_keys):
        raise RuntimeError("Phase-0 evaluation did not produce every expected clip record")
    summary = summarize_records(records, metric_names, decode_cfg["bootstrap"])
    summary.update({
        "phase": "cosmos_vae_ceiling",
        "manifest": str(Path(args.manifest).resolve()),
        "cosmos_model_id": decode_cfg["cosmos"]["model_id"],
        "cosmos_revision": decode_cfg["cosmos"]["revision"],
        "num_frames": int(cfg["data"]["num_frames"]),
        "crop_size": int(cfg["model"]["crop_size"]),
    })
    csv_path = output_dir / decode_cfg["artifacts"]["phase0_records_csv"]
    summary_path = output_dir / decode_cfg["artifacts"]["phase0_metrics"]
    write_csv(csv_path, records)
    atomic_json(summary_path, summary)
    log_metrics(
        wandb_run,
        {f"phase0/summary/{name}": float(interval["mean"])
         for name, interval in summary["metrics"].items()},
    )
    log_artifact(wandb_run, "factorjepa-phase0-records-jsonl", str(records_path), "evaluation")
    log_artifact(wandb_run, "factorjepa-phase0-records-csv", str(csv_path), "evaluation")
    log_artifact(wandb_run, "factorjepa-phase0-summary", str(summary_path), "evaluation")
    print(json.dumps(summary, indent=2))
    finish_wandb(wandb_run)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"FATAL (phase0-cosmos-vae-ceiling): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
