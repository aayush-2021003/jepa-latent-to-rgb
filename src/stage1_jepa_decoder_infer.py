import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_ENDPOINT"] = os.environ.get("FACTORJEPA_HF_ENDPOINT", "https://huggingface.co")
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = os.environ["HF_ENDPOINT"]

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from stage1_jepa_decoder_train import (
    CHECKPOINT_FINAL,
    DEFAULT_SOURCE_HF_FILENAME,
    DEFAULT_SOURCE_HF_REPO,
    Stage1JepaGridDecoder,
    build_full_jepa_grid,
    decode_latents_to_frames,
    export_video,
    load_cosmos_vae,
    load_frozen_jepa,
    resolve_source_ckpt,
    torch_load_cpu,
)
from utils.config import check_gpu, load_merged_config
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.training import build_mask_generators
from utils.video_io import decode_video_bytes


def resize_center_crop(video_tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    video = video_tensor.float() / 255.0
    _, _, h, w = video.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    video = video[:, :, top:top + side, left:left + side]
    video = F.interpolate(video, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    return video.permute(1, 0, 2, 3).contiguous()


def load_clip_bytes_from_local_data(local_data: str, clip_key: str) -> tuple[str, bytes]:
    clip_q, stop_event, reader = iter_clips_parallel(local_data, subset_keys={clip_key})
    try:
        while True:
            item = clip_q.get(timeout=120)
            if item is None:
                break
            found_key, mp4_bytes = item
            if found_key == clip_key:
                return found_key, mp4_bytes
    finally:
        stop_event.set()
        reader.join(timeout=5)
    raise FileNotFoundError(f"clip key not found: {clip_key}")


def load_input_clip(args, cfg: dict) -> tuple[str, torch.Tensor]:
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]
    with tempfile.TemporaryDirectory(prefix="stage1_infer_decode_") as tmp_dir:
        if args.input_mp4:
            path = Path(args.input_mp4)
            if not path.exists():
                raise FileNotFoundError(path)
            clip_key = path.stem
            mp4_bytes = path.read_bytes()
        else:
            clip_key, mp4_bytes = load_clip_bytes_from_local_data(args.local_data, args.clip_key)
        decoded = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames)
        if decoded is None:
            raise RuntimeError(f"failed to decode {clip_key}")
        raw_clip = resize_center_crop(decoded, crop_size)
    return clip_key, raw_clip.unsqueeze(0)


def load_decoder(decoder_ckpt: Path, device) -> tuple[Stage1JepaGridDecoder, dict]:
    ckpt = torch_load_cpu(decoder_ckpt, weights_only=False)
    model_cfg = ckpt["model_cfg"]
    data_cfg = ckpt["data_cfg"]
    stage_cfg = ckpt["stage_cfg"]
    decoder = Stage1JepaGridDecoder(
        embed_dim=model_cfg["embed_dim"],
        n_levels=model_cfg["n_output_distillation"],
        token_grid=tuple(ckpt["token_grid"]),
        latent_shape=tuple(ckpt["latent_shape"]),
        decoder_dim=stage_cfg["decoder_dim"],
        decoder_depth=stage_cfg["decoder_depth"],
        kernel_size=stage_cfg["decoder_kernel_size"],
        dropout=stage_cfg["dropout"],
    ).to(device)
    decoder_state = ckpt.pop("decoder_state_dict")
    decoder.load_state_dict(decoder_state)
    del decoder_state
    ckpt.pop("optimizer", None)
    ckpt.pop("scheduler", None)
    gc.collect()
    decoder.eval()
    decoder.requires_grad_(False)
    return decoder, ckpt


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Infer video with Stage-1 JEPA-grid decoder")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--decoder-ckpt", required=True)
    parser.add_argument("--source-ckpt", default=None)
    parser.add_argument("--source-hf-repo", default=DEFAULT_SOURCE_HF_REPO)
    parser.add_argument("--source-hf-repo-type", default="dataset")
    parser.add_argument("--source-hf-filename", default=DEFAULT_SOURCE_HF_FILENAME)
    parser.add_argument("--input-mp4", default=None)
    parser.add_argument("--local-data", default=None)
    parser.add_argument("--clip-key", default=None)
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    if bool(args.input_mp4) == bool(args.clip_key):
        raise ValueError("Pass exactly one of --input-mp4 or --clip-key")
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.clip_key:
        args.SANITY = False
        args.subset = None
        args.max_tar_files = None
        args.local_data = ensure_local_data(args)

    check_gpu()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    cfg = load_merged_config(args.model_config, args.train_config)
    decoder, ckpt = load_decoder(Path(args.decoder_ckpt), device)
    stage_cfg = ckpt.get("stage_cfg", {})
    pipe = load_cosmos_vae(
        ckpt["cosmos_model_id"],
        ckpt["cosmos_revision"],
        dtype,
        device,
        vae_subfolder=stage_cfg.get("cosmos_vae_subfolder", "vae"),
        enable_tiling=stage_cfg.get("cosmos_vae_enable_tiling", False),
        enable_slicing=stage_cfg.get("cosmos_vae_enable_slicing", False),
    )
    source_ckpt = resolve_source_ckpt(args)
    student, predictor = load_frozen_jepa(
        Path(source_ckpt), cfg["model"], cfg["data"], device, dtype=dtype)
    mask_generators = build_mask_generators(cfg)
    clip_key, raw_batch = load_input_clip(args, cfg)
    raw_batch = raw_batch.to(device)

    full_grid, token_types, grid_meta = build_full_jepa_grid(
        raw_batch, cfg, mask_generators, student, predictor, device)
    pred_latent = decoder(full_grid, token_types)
    frames = decode_latents_to_frames(pipe, pred_latent, dtype, cfg["data"]["num_frames"])

    output_mp4 = Path(args.output_mp4)
    export_video(frames, output_mp4, args.fps)
    output_mp4.with_suffix(".json").write_text(json.dumps({
        "clip_key": clip_key,
        "decoder_ckpt": args.decoder_ckpt,
        "source_ckpt": source_ckpt,
        "jepa_grid": grid_meta,
        "pred_latent_shape": list(pred_latent.shape),
        "fps": args.fps,
        "seed": args.seed,
    }, indent=2) + "\n")
    print(f"Saved video: {output_mp4}")


if __name__ == "__main__":
    main()
