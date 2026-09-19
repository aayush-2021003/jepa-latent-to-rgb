"""Infer pixels from V-JEPA / FactorJEPA predictor latents with a trained Cosmos adapter.

This is the inference sibling of `src/cosmos_train.py`. It reconstructs a visual
clip by:
  1. loading a raw WalkIndia clip,
  2. producing frozen V-JEPA/FactorJEPA predictor latents with the same masking path
     used during training,
  3. projecting those JEPA latents into Cosmos cross-attention space,
  4. running a rectified-flow denoising loop in Cosmos VAE latent space,
  5. decoding the final latent to pixels and exporting an MP4.

Example:
  python -u src/cosmos_infer.py \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/pretrain_encoder.yaml \
    --source-ckpt outputs/full/vjepa_2_1_vitG/train/m09a_pretrain_encoder/m09a_ckpt_best.pt \
    --decoder-ckpt outputs/full/cosmos_decoder/cosmos_latent_decoder_lora_and_projector.pt \
    --input-mp4 /path/to/clip.mp4 \
    --output-mp4 outputs/full/cosmos_decoder/demo.mp4 \
    --num-inference-steps 36 \
    --seed 1
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Avoid huggingface_hub/Xet crashes on Cosmos guardrail/blocklist downloads.
# Must be set before any module imports huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import check_gpu, load_merged_config
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.progress import make_pbar
from utils.training import build_mask_generators, build_student_predictor
from utils.video_io import decode_video_bytes


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
DEFAULT_SOURCE_HF_REPO = "anonymousML123/factorjepa-pretrain-vjepa21-vitG-2B-poc"
DEFAULT_SOURCE_HF_FILENAME = "m09a_ckpt_best.pt"


class CondProjector(nn.Module):
    def __init__(self, jepa_dim: int, cosmos_dim: int, hidden_mult: float = 2.0,
                 num_context_tokens: int = 98):
        super().__init__()
        self.num_context_tokens = num_context_tokens
        hidden = int(jepa_dim * hidden_mult)
        self.net = nn.Sequential(
            nn.LayerNorm(jepa_dim),
            nn.Linear(jepa_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, cosmos_dim),
            nn.LayerNorm(cosmos_dim),
        )

    def forward(self, jepa_latents: torch.Tensor) -> torch.Tensor:
        if jepa_latents.ndim == 3:
            jepa_latents = F.adaptive_avg_pool1d(
                jepa_latents.transpose(1, 2),
                self.num_context_tokens,
            ).transpose(1, 2)
        elif jepa_latents.ndim == 2:
            jepa_latents = jepa_latents.unsqueeze(1).expand(-1, self.num_context_tokens, -1)
        else:
            raise RuntimeError(f"expected JEPA latents with 2 or 3 dims, got {tuple(jepa_latents.shape)}")
        return self.net(jepa_latents)


def infer_cosmos_context_dim_from_dit(dit, fallback: int, bypass_crossattn_proj: bool) -> int:
    crossattn_proj = getattr(dit, "crossattn_proj", None)
    if crossattn_proj is not None:
        for module in crossattn_proj.modules():
            if isinstance(module, nn.Linear):
                return int(module.out_features if bypass_crossattn_proj else module.in_features)
    if hasattr(dit.config, "text_embed_dim"):
        return int(dit.config.text_embed_dim)
    return int(fallback)


def configure_cosmos_conditioning(dit, bypass_crossattn_proj: bool):
    if bypass_crossattn_proj:
        dit.crossattn_proj = nn.Identity()


def select_projector_jepa_latents(jepa_latents: torch.Tensor, model_cfg: dict) -> torch.Tensor:
    embed_dim = model_cfg["embed_dim"]
    n_levels = model_cfg["n_output_distillation"]
    if jepa_latents.shape[-1] == embed_dim:
        return jepa_latents
    if n_levels > 1 and jepa_latents.shape[-1] == embed_dim * n_levels:
        return jepa_latents[..., -embed_dim:]
    raise RuntimeError(
        f"unexpected JEPA latent width {jepa_latents.shape[-1]} "
        f"(expected {embed_dim} or {embed_dim * n_levels})"
    )


def load_cosmos_pipeline(model_id: str, revision: str, dtype: torch.dtype, device, hf_token: str):
    from diffusers import Cosmos2_5_PredictBasePipeline

    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        token=hf_token or None,
    )
    pipe.to(device)
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    return pipe


def add_lora_to_dit(dit, lora_rank: int, lora_alpha: int, use_dora: bool):
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        use_dora=use_dora,
    )
    dit.add_adapter(lora_config)
    cast_training_params(dit, dtype=torch.float32)
    return dit


def normalize_for_jepa(raw_batch_unit_range: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(raw_batch_unit_range.device)
    std = IMAGENET_STD.to(raw_batch_unit_range.device)
    return (raw_batch_unit_range - mean) / std


def resize_center_crop(video_tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Convert decoded uint8 (T,C,H,W) to float (C,T,S,S) in [0,1]."""
    video = video_tensor.float() / 255.0
    _, _, h, w = video.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    video = video[:, :, top:top + side, left:left + side]
    video = F.interpolate(video, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    return video.permute(1, 0, 2, 3).contiguous()


def load_input_clip(args, cfg: dict) -> tuple[str, torch.Tensor]:
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]
    with tempfile.TemporaryDirectory(prefix="cosmos_infer_decode_") as tmp_dir:
        if args.input_mp4:
            clip_path = Path(args.input_mp4)
            if not clip_path.exists():
                raise FileNotFoundError(f"--input-mp4 not found: {clip_path}")
            clip_key = clip_path.stem
            mp4_bytes = clip_path.read_bytes()
        else:
            if not args.local_data or not args.clip_key:
                raise ValueError("Pass either --input-mp4, or both --local-data and --clip-key.")
            clip_key, mp4_bytes = load_clip_bytes_from_local_data(args.local_data, args.clip_key)

        decoded = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames)
        if decoded is None:
            raise RuntimeError(f"failed to decode input clip: {clip_key}")
        raw_clip = resize_center_crop(decoded, crop_size)
    return clip_key, raw_clip.unsqueeze(0)


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
    raise FileNotFoundError(f"clip_key not found in local_data: {clip_key}")


def load_frozen_jepa(source_ckpt_path: Path, model_cfg: dict, data_cfg: dict, device):
    if not source_ckpt_path.exists():
        raise FileNotFoundError(f"--source-ckpt not found: {source_ckpt_path}")
    ckpt = torch.load(source_ckpt_path, map_location="cpu", weights_only=False)
    if "student" not in ckpt or "predictor" not in ckpt:
        raise KeyError(
            f"{source_ckpt_path} must contain both 'student' and 'predictor' keys; "
            f"found {sorted(ckpt.keys())}"
        )
    student, predictor = build_student_predictor(model_cfg, data_cfg)
    student.load_state_dict(ckpt["student"], strict=False)
    predictor.load_state_dict(ckpt["predictor"], strict=False)
    student = student.to(device).eval()
    predictor = predictor.to(device).eval()
    student.requires_grad_(False)
    predictor.requires_grad_(False)
    if hasattr(student, "return_hierarchical"):
        student.return_hierarchical = model_cfg["predict_all"] or model_cfg["n_output_distillation"] > 1
    return student, predictor


def resolve_source_ckpt(args) -> str:
    if args.source_ckpt:
        return args.source_ckpt
    from huggingface_hub import hf_hub_download
    print(f"Downloading source V-JEPA/FactorJEPA checkpoint from HF: "
          f"{args.source_hf_repo}/{args.source_hf_filename}")
    return hf_hub_download(
        repo_id=args.source_hf_repo,
        filename=args.source_hf_filename,
        token=os.environ.get("HF_TOKEN") or None,
    )


def load_decoder_adapter(decoder_ckpt: Path, model_cfg: dict, pipe, device):
    if not decoder_ckpt.exists():
        raise FileNotFoundError(f"--decoder-ckpt not found: {decoder_ckpt}")
    ckpt = torch.load(decoder_ckpt, map_location="cpu", weights_only=False)

    lora_rank = int(ckpt["lora_rank"])
    lora_alpha = int(ckpt["lora_alpha"])
    use_dora = bool(ckpt["use_dora"])
    bypass_crossattn_proj = bool(ckpt.get("bypass_cosmos_crossattn_proj", False))
    configure_cosmos_conditioning(pipe.transformer, bypass_crossattn_proj)
    add_lora_to_dit(pipe.transformer, lora_rank, lora_alpha, use_dora)
    missing, unexpected = pipe.transformer.load_state_dict(ckpt["dit_lora_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected Cosmos LoRA keys: {unexpected[:10]}")
    n_lora_missing = sum(1 for k in missing if "lora" in k.lower())
    if n_lora_missing:
        raise RuntimeError(f"missing {n_lora_missing} LoRA keys while loading {decoder_ckpt}")

    cosmos_dim = infer_cross_attn_dim(pipe, ckpt)
    context_tokens = int(ckpt.get("context_tokens", 1))
    projector = CondProjector(
        jepa_dim=model_cfg["embed_dim"],
        cosmos_dim=cosmos_dim,
        hidden_mult=infer_projector_hidden_mult(ckpt, model_cfg["embed_dim"], cosmos_dim),
        num_context_tokens=context_tokens,
    ).to(device=device, dtype=torch.float32)
    projector.load_state_dict(ckpt["cond_projector_state_dict"])
    projector.eval()
    projector.requires_grad_(False)
    pipe.transformer.eval()
    pipe.transformer.requires_grad_(False)
    return projector, ckpt


def infer_cross_attn_dim(pipe, ckpt: dict) -> int:
    if "cosmos_context_dim" in ckpt:
        return int(ckpt["cosmos_context_dim"])
    if "cosmos_condition_dim" in ckpt:
        return int(ckpt["cosmos_condition_dim"])
    bypass_crossattn_proj = bool(ckpt.get("bypass_cosmos_crossattn_proj", False))
    inferred = infer_cosmos_context_dim_from_dit(
        pipe.transformer,
        int(ckpt["cond_projector_state_dict"]["net.3.weight"].shape[0]),
        bypass_crossattn_proj,
    )
    if inferred == int(ckpt["cond_projector_state_dict"]["net.3.weight"].shape[0]):
        return inferred
    return int(ckpt["cond_projector_state_dict"]["net.3.weight"].shape[0])


def infer_projector_hidden_mult(ckpt: dict, jepa_dim: int, cosmos_dim: int) -> float:
    state = ckpt["cond_projector_state_dict"]
    hidden = state["net.1.weight"].shape[0]
    if state["net.1.weight"].shape[1] != jepa_dim or state["net.3.weight"].shape[0] != cosmos_dim:
        raise RuntimeError("saved CondProjector shape does not match model/cosmos dimensions")
    return hidden / float(jepa_dim)


@torch.no_grad()
def compute_jepa_condition(raw_batch: torch.Tensor, cfg: dict, student, predictor, projector, device) -> torch.Tensor:
    enc_batch = normalize_for_jepa(raw_batch.to(device))
    mp_cfg = cfg["mixed_precision"]
    jepa_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[mp_cfg["dtype"]]
    mask_generators = build_mask_generators(cfg)
    with torch.amp.autocast("cuda", dtype=jepa_dtype, enabled=mp_cfg["enabled"]):
        m_enc, m_pred = mask_generators[0](raw_batch.shape[0])
        m_enc, m_pred = m_enc.to(device), m_pred.to(device)
        n_levels = cfg["model"]["n_output_distillation"]
        z = student(
            enc_batch,
            masks=[m_enc],
            **({"training": True} if n_levels > 1 else {}),
        )
        out = predictor(
            z,
            [m_enc],
            [m_pred],
            **({"mod": "video", "mask_index": 0} if n_levels > 1 else {}),
        )
        jepa_latents = (out[0] if isinstance(out, tuple) else out).float()
        jepa_latents = select_projector_jepa_latents(jepa_latents, cfg["model"])
    return projector(jepa_latents)


@torch.no_grad()
def sample_video_latents(pipe, cond_embeds: torch.Tensor, latent_shape: torch.Size,
                         num_steps: int, dtype: torch.dtype, device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    batch_size, _, num_latent_frames, latent_h, latent_w = latent_shape
    if cond_embeds.ndim != 3 or cond_embeds.shape[0] != batch_size:
        raise RuntimeError(
            f"Cosmos conditioning must be [B, context_tokens, context_dim], got {tuple(cond_embeds.shape)}"
        )
    cond_mask = torch.zeros(batch_size, 1, num_latent_frames, latent_h, latent_w, dtype=dtype, device=device)
    padding_mask = torch.zeros(batch_size, 1, latent_h, latent_w, dtype=dtype, device=device)

    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    pbar = make_pbar(total=num_steps, desc="cosmos_latent_sample", unit="step")
    for i, timestep in enumerate(timesteps):
        sigma = pipe.scheduler.sigmas[i].expand(batch_size).to(device=device, dtype=torch.float32)
        with torch.amp.autocast("cuda", dtype=dtype):
            velocity = pipe.transformer(
                hidden_states=x,
                condition_mask=cond_mask,
                padding_mask=padding_mask,
                timestep=sigma,
                encoder_hidden_states=cond_embeds.to(dtype),
                return_dict=False,
            )[0]
        x = pipe.scheduler.step(velocity, timestep, x, return_dict=False)[0]
        pbar.update(1)
    pbar.close()
    return x


@torch.no_grad()
def decode_latents_to_frames(pipe, latents: torch.Tensor, dtype: torch.dtype) -> list:
    latents_mean = pipe.latents_mean.to(latents.device, latents.dtype)
    latents_std = pipe.latents_std.to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    with torch.amp.autocast("cuda", dtype=dtype):
        decoded = pipe.vae.decode(latents.to(dtype), return_dict=False)[0]
    decoded = pipe._match_num_frames(decoded, pipe._cosmos_infer_num_frames)
    decoded = decoded.float().clamp(-1, 1)
    video = ((decoded[0] + 1.0) * 127.5).clamp(0, 255).byte()
    video = video.permute(1, 2, 3, 0).cpu().numpy()

    from PIL import Image

    return [Image.fromarray(frame) for frame in video]


def export_outputs(frames: list, output_mp4: Path, fps: int, metadata: dict):
    from diffusers.utils import export_to_video

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_mp4), fps=fps)
    meta_path = output_mp4.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved video: {output_mp4}")
    print(f"Saved metadata: {meta_path}")


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(
        description="Generate a pixel-space visualization from V-JEPA/FactorJEPA predictor latents "
                    "using the Cosmos LoRA/projector checkpoint produced by cosmos_train.py."
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--source-ckpt", default=None,
                        help="Optional local combined V-JEPA/FactorJEPA checkpoint with 'student' and "
                             "'predictor'. If omitted, downloads from --source-hf-repo.")
    parser.add_argument("--source-hf-repo", default=DEFAULT_SOURCE_HF_REPO,
                        help="HF model repo containing the FactorJEPA/V-JEPA checkpoint.")
    parser.add_argument("--source-hf-filename", default=DEFAULT_SOURCE_HF_FILENAME,
                        help="Checkpoint filename inside --source-hf-repo.")
    parser.add_argument("--decoder-ckpt", required=True,
                        help="cosmos_latent_decoder_lora_and_projector.pt from cosmos_train.py.")
    parser.add_argument("--input-mp4", default=None,
                        help="Local MP4 to condition on. Mutually exclusive with --clip-key.")
    parser.add_argument("--local-data", default=None,
                        help="Local WalkIndia shard dir, used with --clip-key.")
    parser.add_argument("--clip-key", default=None,
                        help="Clip key inside --local-data to condition on.")
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=36)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--height", type=int, default=None,
                        help="Generated video height. Defaults to cfg.data.crop_size.")
    parser.add_argument("--width", type=int, default=None,
                        help="Generated video width. Defaults to cfg.data.crop_size.")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    if bool(args.input_mp4) == bool(args.clip_key):
        raise ValueError("Pass exactly one input source: --input-mp4 OR --clip-key with --local-data.")
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    check_gpu()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = load_merged_config(args.model_config, args.train_config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    decoder_meta = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    model_id = decoder_meta["cosmos_model_id"]
    revision = decoder_meta["cosmos_revision"]
    if args.clip_key:
        args.SANITY = False
        args.subset = None
        args.local_data = ensure_local_data(args)
    source_ckpt = resolve_source_ckpt(args)

    clip_key, raw_batch = load_input_clip(args, cfg)
    student, predictor = load_frozen_jepa(Path(source_ckpt), model_cfg, data_cfg, device)
    pipe = load_cosmos_pipeline(model_id, revision, dtype, device, os.environ.get("HF_TOKEN", ""))
    projector, adapter_meta = load_decoder_adapter(Path(args.decoder_ckpt), model_cfg, pipe, device)

    cond_embeds = compute_jepa_condition(raw_batch, cfg, student, predictor, projector, device)
    height = args.height or int(data_cfg["crop_size"])
    width = args.width or int(data_cfg["crop_size"])
    num_frames = int(data_cfg["num_frames"])
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"--height and --width must be divisible by 16, got {height}x{width}")
    num_channels_latents = int(pipe.transformer.config.in_channels) - 1
    latent_shape = (
        1,
        num_channels_latents,
        (num_frames - 1) // pipe.vae_scale_factor_temporal + 1,
        height // pipe.vae_scale_factor_spatial,
        width // pipe.vae_scale_factor_spatial,
    )
    pipe._cosmos_infer_num_frames = num_frames
    latents = sample_video_latents(
        pipe=pipe,
        cond_embeds=cond_embeds,
        latent_shape=latent_shape,
        num_steps=args.num_inference_steps,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )
    frames = decode_latents_to_frames(pipe, latents, dtype)

    metadata = {
        "clip_key": clip_key,
        "input_mp4": args.input_mp4,
        "local_data": args.local_data,
        "source_ckpt": source_ckpt,
        "source_hf_repo": args.source_hf_repo,
        "source_hf_filename": args.source_hf_filename,
        "decoder_ckpt": args.decoder_ckpt,
        "cosmos_model_id": model_id,
        "cosmos_revision": revision,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "fps": args.fps,
        "num_frames": len(frames),
        "latent_shape": list(latent_shape),
        "jepa_condition_shape": list(cond_embeds.shape),
        "lora_rank": adapter_meta["lora_rank"],
        "lora_alpha": adapter_meta["lora_alpha"],
        "use_dora": adapter_meta["use_dora"],
    }
    export_outputs(frames, Path(args.output_mp4), args.fps, metadata)


if __name__ == "__main__":
    main()
