import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_ENDPOINT"] = os.environ.get("FACTORJEPA_HF_ENDPOINT", "https://huggingface.co")
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = os.environ["HF_ENDPOINT"]

import torch

sys.path.insert(0, str(Path(__file__).parent))

import stage1_jepa_decoder_train as stage1_train
from stage1_jepa_decoder_infer import load_decoder as load_stage1_decoder
from stage1_jepa_decoder_train import (
    DEFAULT_SOURCE_HF_FILENAME,
    DEFAULT_SOURCE_HF_REPO,
    DEFAULT_SOURCE_CKPT,
    build_full_jepa_grid,
    decode_latents_to_frames,
    encode_video_latents,
    load_cosmos_vae,
    load_frozen_jepa,
    raw_batch_to_frames,
    sample_fixed_validation_masks,
    torch_load_cpu,
)
from stage2_latent_refiner_train import (
    Stage2LatentRefiner,
    export_video,
    load_input_sample,
)
from utils.config import check_gpu, load_merged_config
from utils.data_download import ensure_local_data
from utils.training import build_mask_generators


def load_refiner(stage2_ckpt: Path, device) -> tuple[Stage2LatentRefiner, dict]:
    payload = torch_load_cpu(stage2_ckpt, weights_only=False)
    stage2_cfg = payload["stage2_cfg"]
    model_cfg = payload["model_cfg"]
    latent_shape = tuple(payload["latent_shape"])
    refiner = Stage2LatentRefiner(
        latent_channels=latent_shape[0],
        embed_dim=model_cfg["embed_dim"],
        n_levels=model_cfg["n_output_distillation"],
        hidden_dim=stage2_cfg["hidden_dim"],
        depth=stage2_cfg["depth"],
        kernel_size=stage2_cfg["kernel_size"],
        dropout=stage2_cfg["dropout"],
        cond_dim=stage2_cfg["cond_dim"],
        residual_scale=stage2_cfg["residual_scale"],
    ).to(device)
    refiner_state = payload.pop("refiner_state_dict")
    refiner.load_state_dict(refiner_state)
    del refiner_state
    payload.pop("optimizer", None)
    payload.pop("scheduler", None)
    gc.collect()
    refiner.eval()
    refiner.requires_grad_(False)
    return refiner, payload


def resolve_source_ckpt(args, payload: dict) -> str:
    if args.source_ckpt and str(args.source_ckpt).lower() not in {"", "none", "null"}:
        return args.source_ckpt
    payload_source = payload.get("source_ckpt")
    if payload_source and Path(payload_source).exists():
        return payload_source
    from huggingface_hub import hf_hub_download
    print(f"Downloading source V-JEPA/FactorJEPA checkpoint from HF: "
          f"{args.source_hf_repo}/{args.source_hf_filename}")
    return hf_hub_download(
        repo_id=args.source_hf_repo,
        filename=args.source_hf_filename,
        repo_type=getattr(args, "source_hf_repo_type", "dataset"),
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
    )


def save_optional_video(frames: list, path_str: str | None, fps: int) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    export_video(frames, path, fps)
    return str(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Infer video with Stage-1 decoder + Stage-2 latent refiner")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--stage1-ckpt", default=None,
                        help="Defaults to the Stage-1 checkpoint path saved inside --stage2-ckpt.")
    parser.add_argument("--source-ckpt", default=DEFAULT_SOURCE_CKPT)
    parser.add_argument("--source-hf-repo", default=DEFAULT_SOURCE_HF_REPO)
    parser.add_argument("--source-hf-repo-type", default="dataset")
    parser.add_argument("--source-hf-filename", default=DEFAULT_SOURCE_HF_FILENAME)
    parser.add_argument("--input-mp4", default=None)
    parser.add_argument("--local-data", default=None)
    parser.add_argument("--clip-key", default=None)
    parser.add_argument("--max-tar-files", type=int, default=None)
    parser.add_argument("--output-mp4", required=True,
                        help="Final Stage-2 refined reconstruction.")
    parser.add_argument("--coarse-output-mp4", default=None,
                        help="Optional Stage-1-only reconstruction for comparison.")
    parser.add_argument("--reference-output-mp4", default=None,
                        help="Optional preprocessed input/reference clip.")
    parser.add_argument("--target-roundtrip-output-mp4", default=None,
                        help="Optional Cosmos VAE target roundtrip.")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--mask-seed", type=int, default=None)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    if args.input_mp4 and args.clip_key:
        raise ValueError("Pass at most one of --input-mp4 or --clip-key")
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        stage1_train.HF_TOKEN = args.hf_token

    args.SANITY = False
    args.POC = False
    args.FULL = True
    args.subset = None

    check_gpu()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    merged_cfg = load_merged_config(args.model_config, args.train_config)
    refiner, stage2_payload = load_refiner(Path(args.stage2_ckpt), device)
    cfg = dict(merged_cfg)
    cfg["model"] = stage2_payload["model_cfg"]
    cfg["data"] = stage2_payload["data_cfg"]

    stage2_cfg = stage2_payload["stage2_cfg"]
    mask_seed = args.mask_seed
    if mask_seed is None:
        mask_seed = stage2_cfg.get("overfit", {}).get("fixed_mask_seed", 1)
    torch.manual_seed(mask_seed)
    torch.cuda.manual_seed_all(mask_seed)

    if args.max_tar_files is None:
        args.max_tar_files = stage2_cfg.get("overfit", {}).get("max_tar_files", 1)
    if not args.input_mp4:
        args.local_data = ensure_local_data(args)

    stage1_ckpt = args.stage1_ckpt or stage2_payload.get("stage1_ckpt")
    if not stage1_ckpt:
        raise ValueError("Stage-1 checkpoint not provided and not found inside Stage-2 checkpoint")

    stage1_decoder, stage1_payload = load_stage1_decoder(Path(stage1_ckpt), device)
    stage1_cfg = stage1_payload.get("stage_cfg", {})
    pipe = load_cosmos_vae(
        stage1_payload["cosmos_model_id"],
        stage1_payload["cosmos_revision"],
        dtype,
        device,
        vae_subfolder=stage1_cfg.get("cosmos_vae_subfolder", "vae"),
        enable_tiling=stage1_cfg.get("cosmos_vae_enable_tiling", False),
        enable_slicing=stage1_cfg.get("cosmos_vae_enable_slicing", False),
    )

    source_ckpt = resolve_source_ckpt(args, stage2_payload)
    student, predictor = load_frozen_jepa(
        Path(source_ckpt), cfg["model"], cfg["data"], device, dtype=dtype)
    mask_generators = build_mask_generators(cfg)
    fixed_masks = sample_fixed_validation_masks(mask_generators, batch_size=1, seed=mask_seed)

    clip_key, raw_batch = load_input_sample(args, cfg, args.max_tar_files)
    raw_batch = raw_batch.to(device)
    full_grid, token_types, grid_meta = build_full_jepa_grid(
        raw_batch, cfg, mask_generators, student, predictor, device, fixed_masks=fixed_masks)
    coarse_latent = stage1_decoder(full_grid, token_types)
    refined_latent = refiner(coarse_latent, full_grid, token_types)
    target_latent = encode_video_latents(pipe, raw_batch, dtype)

    output_mp4 = Path(args.output_mp4)
    refined_frames = decode_latents_to_frames(pipe, refined_latent, dtype, cfg["data"]["num_frames"])
    export_video(refined_frames, output_mp4, args.fps)

    saved = {"stage2_refined": str(output_mp4)}
    saved["stage1_coarse"] = save_optional_video(
        decode_latents_to_frames(pipe, coarse_latent, dtype, cfg["data"]["num_frames"]),
        args.coarse_output_mp4,
        args.fps,
    )
    saved["reference_input"] = save_optional_video(
        raw_batch_to_frames(raw_batch.detach().cpu()),
        args.reference_output_mp4,
        args.fps,
    )
    saved["target_vae_roundtrip"] = save_optional_video(
        decode_latents_to_frames(pipe, target_latent, dtype, cfg["data"]["num_frames"]),
        args.target_roundtrip_output_mp4,
        args.fps,
    )

    output_mp4.with_suffix(".json").write_text(json.dumps({
        "clip_key": clip_key,
        "stage1_ckpt": stage1_ckpt,
        "stage2_ckpt": args.stage2_ckpt,
        "source_ckpt": source_ckpt,
        "mask_seed": mask_seed,
        "jepa_grid": grid_meta,
        "coarse_latent_shape": list(coarse_latent.shape),
        "refined_latent_shape": list(refined_latent.shape),
        "target_latent_shape": list(target_latent.shape),
        "saved": saved,
    }, indent=2) + "\n")
    print(f"Saved Stage-2 refined video: {output_mp4}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback
        print(f"\nFATAL (stage2-latent-refiner-infer): {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
