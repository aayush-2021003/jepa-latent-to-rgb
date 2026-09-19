"""Phase 2: decode predictor-substituted FactorJEPA features with a frozen oracle decoder.

Official architecture references:
  https://github.com/facebookresearch/vjepa2
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py

The decoder is never fine-tuned on predictor outputs. For each official masking
regime, this script starts from the full oracle encoder grid and replaces only
target positions with FactorJEPA predictor outputs. That isolates predictor error
from the decoder's own reconstruction error.

USAGE
  python -u src/predicted_jepa_decoder_eval.py \
    --model-config configs/model/vjepa2_1_vitg.yaml \
    --train-config configs/train/surgery_3stage_DI_diheavy_encoder.yaml \
    --decode-config configs/jepa_pixel_decode.yaml \
    --factorjepa-repo anonymousML123/factorjepa-outputs \
    --factorjepa-repo-type dataset \
    --factorjepa-filename outputs/full/vjepa_2_1_vitg_1B/train/m09c_surgery_3stage_DI_diheavy_encoder/m09c_ckpt_best.pt \
    --oracle-decoder-checkpoint outputs/jepa_pixel_decode/oracle_poc/oracle_decoder_best.pt \
    --local-data data/full_local \
    --test-manifest data/jepa_decode_splits/poc/test.json \
    --output-dir outputs/jepa_pixel_decode/predicted_poc \
    --wandb-group factorjepa-one-sample \
    --cache-policy 2
"""
import argparse
import json
import traceback
from pathlib import Path

import torch

from utils.cache_policy import add_cache_policy_arg, resolve_cache_policy_interactive, wipe_output_dir
from utils.config import load_merged_config
from utils.jepa_pixel_decode import (
    LPIPSComputer,
    append_jsonl,
    atomic_json,
    decode_cosmos_latents,
    encode_cosmos_latents,
    export_comparison_video,
    extract_oracle_features,
    iter_video_batches,
    load_cosmos_vae_only,
    load_factorjepa_checkpoint,
    load_oracle_decoder,
    per_clip_latent_metrics,
    per_clip_pixel_metrics,
    read_clip_manifest,
    read_jsonl_records,
    require_cuda,
    resolve_hf_checkpoint,
    sample_seeded_masks,
    sha256_file,
    stable_clip_seed,
    substitute_predicted_features,
    summarize_records,
    target_mask_to_pixels,
    write_csv,
)
from utils.progress import make_pbar
from utils.training import build_mask_generators, load_config
from utils.wandb_utils import (
    add_wandb_args,
    finish_wandb,
    init_wandb,
    log_artifact,
    log_metrics,
    log_video,
)


def add_metric_group(record: dict, prefix: str, metrics: dict[str, torch.Tensor],
                     batch_index: int):
    record.update({
        f"{prefix}_{name}": float(values[batch_index].detach())
        for name, values in metrics.items()
    })


def assert_checkpoint_compatible(payload: dict, cfg: dict, decode_cfg: dict,
                                 source_checkpoint: str):
    supported_contracts = {
        "oracle_full_encoder_features_to_cosmos_vae_latent",
        "one_sample_oracle_memorization_diagnostic",
    }
    if payload["experiment_contract"] not in supported_contracts:
        raise RuntimeError(f"unsupported oracle experiment contract: {payload['experiment_contract']}")
    if payload["model_cfg"] != cfg["model"] or payload["data_cfg"] != cfg["data"]:
        raise RuntimeError("oracle decoder FactorJEPA model/data config differs from evaluation config")
    if payload["decode_cfg"]["decoder"] != decode_cfg["decoder"]:
        raise RuntimeError("oracle decoder architecture config differs from evaluation config")
    if payload["decode_cfg"]["cosmos"] != decode_cfg["cosmos"]:
        raise RuntimeError("oracle decoder Cosmos VAE config differs from evaluation config")
    if payload["decode_cfg"]["factorjepa"] != decode_cfg["factorjepa"]:
        raise RuntimeError("oracle decoder FactorJEPA precision config differs from evaluation config")
    if int(payload["source_checkpoint_size_bytes"]) != Path(source_checkpoint).stat().st_size:
        raise RuntimeError("oracle decoder was trained against a different FactorJEPA file size")
    current_sha256 = sha256_file(
        source_checkpoint, int(decode_cfg["provenance"]["sha256_chunk_size_bytes"]))
    if payload["source_checkpoint_sha256"] != current_sha256:
        raise RuntimeError("oracle decoder was trained against a different FactorJEPA SHA-256")


def target_feature_metrics(predicted: torch.Tensor, oracle: torch.Tensor,
                           target_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_indices = torch.arange(
        oracle.shape[0], device=oracle.device).unsqueeze(1).expand_as(target_mask)
    target_oracle = oracle[batch_indices, target_mask]
    difference = predicted.float() - target_oracle.float()
    return {
        "feature_mse": difference.pow(2).flatten(1).mean(dim=1),
        "feature_mae": difference.abs().flatten(1).mean(dim=1),
    }


def main():
    parser = argparse.ArgumentParser("Evaluate predicted FactorJEPA latents through the oracle decoder")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--decode-config", required=True)
    parser.add_argument("--factorjepa-repo", required=True)
    parser.add_argument("--factorjepa-repo-type", required=True)
    parser.add_argument("--factorjepa-filename", required=True)
    parser.add_argument("--oracle-decoder-checkpoint", required=True)
    parser.add_argument("--local-data", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-group", required=True)
    add_cache_policy_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    cfg = load_merged_config(args.model_config, args.train_config)
    decode_cfg = load_config(args.decode_config)
    if int(decode_cfg["evaluation"]["batch_size"]) != 1:
        raise ValueError("predicted evaluation requires batch_size=1 for per-clip deterministic masks")
    device = require_cuda()
    output_dir = Path(args.output_dir)
    wipe_output_dir(output_dir, args.cache_policy, label="predicted JEPA evaluation output")
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(
        module_name="predicted_jepa_decoder_eval",
        mode="EVAL",
        config={
            "phase": "phase2",
            "test_manifest": args.test_manifest,
            "model_config": args.model_config,
            "train_config": args.train_config,
            "decode_config": args.decode_config,
            "factorjepa_filename": args.factorjepa_filename,
            "oracle_decoder_checkpoint": args.oracle_decoder_checkpoint,
        },
        enabled=not args.no_wandb,
        group=args.wandb_group,
    )

    test_keys = read_clip_manifest(args.test_manifest)
    source_checkpoint = resolve_hf_checkpoint(
        args.factorjepa_repo, args.factorjepa_repo_type, args.factorjepa_filename)
    student, predictor, load_report = load_factorjepa_checkpoint(
        checkpoint_path=source_checkpoint,
        model_cfg=cfg["model"],
        data_cfg=cfg["data"],
        load_predictor=True,
        device=device,
        factor_cfg=decode_cfg["factorjepa"],
    )
    decoder, oracle_payload = load_oracle_decoder(args.oracle_decoder_checkpoint, device)
    assert_checkpoint_compatible(oracle_payload, cfg, decode_cfg, source_checkpoint)
    cosmos = load_cosmos_vae_only(decode_cfg["cosmos"], device)
    lpips_computer = LPIPSComputer(decode_cfg["metrics"], device)
    mask_generators = build_mask_generators(cfg)
    mask_names = list(decode_cfg["evaluation"]["mask_names"])
    if len(mask_generators) != len(mask_names):
        raise RuntimeError(
            f"configured mask names={len(mask_names)} but FactorJEPA has {len(mask_generators)} masks")

    records_path = output_dir / decode_cfg["artifacts"]["predicted_records"]
    records = read_jsonl_records(records_path, ("clip_key", "mask_name"))
    completed = {(record["clip_key"], record["mask_name"]) for record in records}
    expected = {(clip_key, mask_name) for clip_key in test_keys for mask_name in mask_names}
    unexpected = completed - expected
    if unexpected:
        raise RuntimeError(f"predicted records contain {len(unexpected)} unexpected clip/mask identities")
    remaining_keys = [
        clip_key for clip_key in test_keys
        if any((clip_key, mask_name) not in completed for mask_name in mask_names)
    ]
    gallery_counts = {
        mask_name: sum(record["mask_name"] == mask_name for record in records)
        for mask_name in mask_names
    }
    for mask_name, existing_count in gallery_counts.items():
        upload_count = min(existing_count, int(decode_cfg["evaluation"]["qualitative_clips"]))
        for existing_index in range(upload_count):
            gallery_name = decode_cfg["artifacts"]["predicted_gallery_pattern"].format(
                index=existing_index)
            existing_video_path = (
                output_dir / decode_cfg["artifacts"]["predicted_gallery_dir"] /
                mask_name / gallery_name
            )
            log_video(
                wandb_run,
                f"phase2/reconstruction/{mask_name}",
                str(existing_video_path),
                int(decode_cfg["video"]["fps"]),
            )

    batch_iterator = iter_video_batches(
            local_data=args.local_data,
            clip_keys=remaining_keys,
            num_frames=int(cfg["data"]["num_frames"]),
            crop_size=int(cfg["model"]["crop_size"]),
            batch_size=int(decode_cfg["evaluation"]["batch_size"]),
            num_readers=int(decode_cfg["data_loading"]["tar_readers"]),
            decode_workers=int(decode_cfg["data_loading"]["decode_workers"]),
            queue_timeout_seconds=int(decode_cfg["data_loading"]["queue_timeout_seconds"]),
            reader_join_timeout_seconds=int(
                decode_cfg["data_loading"]["reader_join_timeout_seconds"]),
            training=False,
            augmentation_cfg=cfg["augmentation"]) if remaining_keys else ()
    progress = make_pbar(total=len(remaining_keys), desc="predicted JEPA evaluation", unit="clip")
    for raw_batch, batch_keys in batch_iterator:
        raw_batch = raw_batch.to(device)
        clip_key = batch_keys[0]
        oracle_features = extract_oracle_features(
            raw_batch, student, cfg["model"], decode_cfg["factorjepa"])
        target_latent = encode_cosmos_latents(cosmos, raw_batch)
        oracle_latent = decoder(oracle_features)
        vae_video = decode_cosmos_latents(
            cosmos, target_latent, int(cfg["data"]["num_frames"]), requires_grad=False)
        oracle_video = decode_cosmos_latents(
            cosmos, oracle_latent, int(cfg["data"]["num_frames"]), requires_grad=False)
        common_groups = {
            "oracle_vs_target": per_clip_latent_metrics(oracle_latent, target_latent),
            "vae_vs_input": per_clip_pixel_metrics(
                vae_video, raw_batch, decode_cfg["metrics"], lpips_computer),
            "oracle_vs_input": per_clip_pixel_metrics(
                oracle_video, raw_batch, decode_cfg["metrics"], lpips_computer),
            "oracle_vs_vae": per_clip_pixel_metrics(
                oracle_video, vae_video, decode_cfg["metrics"], lpips_computer),
        }

        for mask_index, (mask_name, mask_generator) in enumerate(zip(mask_names, mask_generators)):
            if (clip_key, mask_name) in completed:
                continue
            seed = stable_clip_seed(int(decode_cfg["evaluation"]["mask_seed"]), clip_key, mask_name)
            context_mask, target_mask = sample_seeded_masks(
                mask_generator, batch_size=raw_batch.shape[0], seed=seed)
            context_mask = context_mask.to(device)
            target_mask = target_mask.to(device)
            if torch.isin(context_mask, target_mask).any():
                raise RuntimeError(f"context/target masks overlap for {clip_key} mask={mask_name}")
            hybrid_features, predicted_features = substitute_predicted_features(
                raw_batch=raw_batch,
                oracle_features=oracle_features,
                student=student,
                predictor=predictor,
                context_mask=context_mask,
                target_mask=target_mask,
                model_cfg=cfg["model"],
                factor_cfg=decode_cfg["factorjepa"],
                mask_index=mask_index,
            )
            predicted_latent = decoder(hybrid_features)
            predicted_video = decode_cosmos_latents(
                cosmos, predicted_latent, int(cfg["data"]["num_frames"]), requires_grad=False)
            pixel_mask = target_mask_to_pixels(target_mask, cfg["model"], cfg["data"])
            feature_metrics = target_feature_metrics(
                predicted_features, oracle_features, target_mask)
            predicted_groups = {
                "predicted_vs_target": per_clip_latent_metrics(predicted_latent, target_latent),
                "predicted_vs_oracle": per_clip_latent_metrics(predicted_latent, oracle_latent),
                "predicted_vs_input": per_clip_pixel_metrics(
                    predicted_video, raw_batch, decode_cfg["metrics"], lpips_computer),
                "predicted_vs_vae": per_clip_pixel_metrics(
                    predicted_video, vae_video, decode_cfg["metrics"], lpips_computer),
                "predicted_vs_oracle_pixel": per_clip_pixel_metrics(
                    predicted_video, oracle_video, decode_cfg["metrics"], lpips_computer),
                "predicted_target_features_vs_oracle": feature_metrics,
            }
            masked_metrics = per_clip_pixel_metrics(
                predicted_video, oracle_video, decode_cfg["metrics"], lpips_computer, mask=pixel_mask)
            record = {
                "clip_key": clip_key,
                "mask_name": mask_name,
                "mask_index": mask_index,
                "mask_seed": seed,
                "context_tokens": int(context_mask.shape[1]),
                "target_tokens": int(target_mask.shape[1]),
                "target_fraction": float(target_mask.shape[1] / oracle_features.shape[1]),
                "context_fraction": float(context_mask.shape[1] / oracle_features.shape[1]),
                "oracle_preserved_noncontext_fraction": float(
                    1.0 - (context_mask.shape[1] + target_mask.shape[1]) / oracle_features.shape[1]),
            }
            for prefix, metrics in common_groups.items():
                add_metric_group(record, prefix, metrics, 0)
            for prefix, metrics in predicted_groups.items():
                add_metric_group(record, prefix, metrics, 0)
            for metric_name in ("pixel_mse", "pixel_mae", "psnr_db"):
                record[f"masked_predicted_vs_oracle_{metric_name}"] = float(
                    masked_metrics[metric_name][0].detach())
            record["prediction_penalty_latent_mae"] = (
                record["predicted_vs_target_latent_mae"] - record["oracle_vs_target_latent_mae"])
            record["prediction_penalty_pixel_mae"] = (
                record["predicted_vs_vae_pixel_mae"] - record["oracle_vs_vae_pixel_mae"])

            wandb_step = len(records)
            has_video = gallery_counts[mask_name] < int(
                decode_cfg["evaluation"]["qualitative_clips"])
            log_metrics(
                wandb_run,
                {f"phase2/per_clip/{mask_name}/{name}": value
                 for name, value in record.items() if isinstance(value, (int, float))},
                step=wandb_step,
                commit=not has_video,
            )
            if has_video:
                gallery_name = decode_cfg["artifacts"]["predicted_gallery_pattern"].format(
                    index=gallery_counts[mask_name])
                video_path = (
                    output_dir / decode_cfg["artifacts"]["predicted_gallery_dir"] /
                    mask_name / gallery_name
                )
                export_comparison_video(
                    [raw_batch[0], vae_video[0], oracle_video[0], predicted_video[0]],
                    ["input", "Cosmos VAE", "oracle JEPA", f"predicted: {mask_name}"],
                    video_path,
                    int(decode_cfg["video"]["fps"]),
                    decode_cfg["video"]["labels"],
                )
                log_video(
                    wandb_run,
                    f"phase2/reconstruction/{mask_name}",
                    str(video_path),
                    int(decode_cfg["video"]["fps"]),
                    step=wandb_step,
                    commit=True,
                )
            append_jsonl(records_path, record)
            records.append(record)
            gallery_counts[mask_name] += 1
        progress.update(len(batch_keys))
    progress.close()

    if {(record["clip_key"], record["mask_name"]) for record in records} != expected:
        raise RuntimeError("predicted evaluation did not produce every expected clip/mask record")
    metadata_fields = {
        "clip_key", "mask_name", "mask_index", "mask_seed", "context_tokens", "target_tokens",
        "target_fraction", "context_fraction", "oracle_preserved_noncontext_fraction",
    }
    per_mask = {}
    for mask_name in mask_names:
        mask_records = [record for record in records if record["mask_name"] == mask_name]
        metric_names = sorted({key for record in mask_records for key in record} - metadata_fields)
        per_mask[mask_name] = summarize_records(mask_records, metric_names, decode_cfg["bootstrap"])
    summary = {
        "phase": "predicted_frozen_oracle_decoder_evaluation",
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "n_test_clips": len(test_keys),
        "mask_seed": int(decode_cfg["evaluation"]["mask_seed"]),
        "mask_names": mask_names,
        "per_mask": per_mask,
        "factorjepa_checkpoint": args.factorjepa_filename,
        "factorjepa_load_report": load_report,
        "oracle_decoder_checkpoint": str(Path(args.oracle_decoder_checkpoint).resolve()),
        "experiment_contract": "oracle_grid_with_target_only_predictor_substitution",
    }
    csv_path = output_dir / decode_cfg["artifacts"]["predicted_records_csv"]
    summary_path = output_dir / decode_cfg["artifacts"]["predicted_metrics"]
    write_csv(csv_path, records)
    atomic_json(summary_path, summary)
    summary_metrics = {}
    for mask_name, mask_summary in per_mask.items():
        summary_metrics.update({
            f"phase2/summary/{mask_name}/{metric_name}": float(interval["mean"])
            for metric_name, interval in mask_summary["metrics"].items()
        })
    log_metrics(wandb_run, summary_metrics)
    log_artifact(wandb_run, "factorjepa-phase2-records-jsonl", str(records_path), "evaluation")
    log_artifact(wandb_run, "factorjepa-phase2-records-csv", str(csv_path), "evaluation")
    log_artifact(wandb_run, "factorjepa-phase2-summary", str(summary_path), "evaluation")
    print(json.dumps(summary, indent=2))
    finish_wandb(wandb_run)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"FATAL (predicted-jepa-decoder): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
