import argparse
import json
import math
import os
import sys
import threading
import queue
import time
from pathlib import Path

# Avoid huggingface_hub/Xet crashes on Cosmos guardrail/blocklist downloads.
# Must be set before any module imports huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

from utils.live_debug import install_debug_handlers
install_debug_handlers()

from utils.config import check_gpu, load_subset, get_pipeline_config, load_merged_config
from utils.gpu_batch import cuda_cleanup, add_gpu_mem_arg
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
from utils.progress import make_pbar
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb
from utils.cache_policy import add_cache_policy_arg, resolve_cache_policy_interactive, wipe_output_dir
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.video_io import decode_video_bytes
from utils.data_paths import artifact
from utils.training import build_student_predictor, build_mask_generators, load_config, atomic_torch_save

_pcfg = get_pipeline_config()
PREFETCH_QUEUE_SIZE = _pcfg["streaming"]["prefetch_queue_train"]
CHECKPOINT_PREFIX = "m14_cosmos_decoder_ckpt"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

# Dataset this script trains against.
WALKINDIA_DATASET_ID = "anonymousML123/walkindia-200k"
DEFAULT_SOURCE_HF_REPO = "anonymousML123/factorjepa-pretrain-vjepa21-vitG-2B-poc"
DEFAULT_SOURCE_HF_FILENAME = "m09a_ckpt_best.pt"


def load_cosmos_pipeline(model_id: str, revision: str, dtype: torch.dtype, device):
    from diffusers import Cosmos2_5_PredictBasePipeline

    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype, token=HF_TOKEN or None,
    )
    pipe.to(device)

    dit = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder

    dit.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    return pipe, dit, vae, text_encoder


def add_lora_to_dit(dit, lora_rank: int, lora_alpha: int, use_dora: bool):
    from peft import LoraConfig
    from diffusers.training_utils import cast_training_params

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        use_dora=use_dora,
    )
    dit.add_adapter(lora_config)
    n_adapter_params = 0
    for name, param in dit.named_parameters():
        lname = name.lower()
        if "lora" in lname or "dora" in lname or "adapter" in lname:
            param.requires_grad_(True)
            n_adapter_params += param.numel()
    if n_adapter_params == 0:
        raise RuntimeError("LoRA adapter insertion produced 0 trainable adapter parameters")
    cast_training_params(dit, dtype=torch.float32)
    return dit


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


def infer_cosmos_context_dim(dit, fallback: int, bypass_crossattn_proj: bool) -> int:
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
    """Use one embed_dim-wide V-JEPA level for Cosmos conditioning.

    V-JEPA 2.1 predictor outputs can be hierarchical concats such as
    6656 = 4 * 1664. The projector is intentionally sized for one token
    stream, so keep the final feature level.
    """
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


def augment_clip_consistent_with_raw_target(video_tensor: torch.Tensor, cfg_aug: dict, crop_size: int):
    import torchvision.transforms as TT

    T_frames, C, H, W = video_tensor.shape
    scale = cfg_aug["random_resize_scale"]
    ratio = cfg_aug["random_resize_ratio"]
    i, j, h, w = TT.RandomResizedCrop.get_params(video_tensor[0], scale=scale, ratio=ratio)

    video = video_tensor.float() / 255.0
    video = video[:, :, i:i + h, j:j + w]
    video = F.interpolate(video, size=(crop_size, crop_size), mode="bilinear", align_corners=False)

    if torch.rand(1).item() < cfg_aug["horizontal_flip"]:
        video = video.flip(-1)

    raw_target = video.clone()
    return raw_target


def normalize_for_jepa(raw_batch_unit_range: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(raw_batch_unit_range.device)
    std = IMAGENET_STD.to(raw_batch_unit_range.device)
    return (raw_batch_unit_range - mean) / std

def decoder_producer_thread(cfg: dict, q: queue.Queue, stop_event: threading.Event,
                             clip_keys: set, local_data: str, max_tar_files: int = None):
    from concurrent.futures import ThreadPoolExecutor
    torch.set_num_threads(1)

    batch_size = cfg["optimization"]["batch_size"]
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]
    cfg_aug = cfg["augmentation"]
    decode_workers = _pcfg["streaming"]["decode_workers_train"]
    max_retries = _pcfg["streaming"]["max_retries"]

    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp(prefix="m14_cosmos_decoder_")
    retries = 0

    def _decode_batch(pool, pending_bytes, pending_keys):
        futures = [pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                   for b, k in zip(pending_bytes, pending_keys)]
        results = [(f.result(), k) for f, k in zip(futures, pending_keys)]
        batch_tensors = [t for t, k in results if t is not None]
        batch_keys = [k for t, k in results if t is not None]
        if not batch_tensors:
            return
        raw_list = [augment_clip_consistent_with_raw_target(vt, cfg_aug, crop_size) for vt in batch_tensors]
        raw_batch = torch.stack(raw_list, dim=0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        q.put(("batch", raw_batch, batch_keys[:]))

    try:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            while not stop_event.is_set():
                try:
                    clip_q, tar_stop, reader = iter_clips_parallel(
                        local_data,
                        subset_keys=clip_keys or None,
                        max_tar_files=max_tar_files,
                    )
                    pending_bytes, pending_keys = [], []
                    try:
                        while not stop_event.is_set():
                            item = clip_q.get(timeout=120)
                            if item is None:
                                break
                            clip_key, mp4_bytes = item
                            if not mp4_bytes:
                                continue
                            pending_bytes.append(mp4_bytes)
                            pending_keys.append(clip_key)
                            if len(pending_bytes) >= batch_size:
                                _decode_batch(pool, pending_bytes, pending_keys)
                                pending_bytes, pending_keys = [], []
                    finally:
                        tar_stop.set()
                        reader.join(timeout=5)
                    if stop_event.is_set():
                        break
                    if pending_bytes and not stop_event.is_set():
                        _decode_batch(pool, pending_bytes, pending_keys)
                except (ConnectionError, TimeoutError, OSError) as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"  FATAL: producer stream failed after {max_retries} retries: {e}")
                        break
                    wait = min(2 ** retries, 60)
                    print(f"  Stream error ({e}), retry {retries}/{max_retries} in {wait}s")
                    time.sleep(wait)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        q.put(("error" if retries > max_retries else "done", None, None))


def load_frozen_jepa(source_ckpt_path: Path, model_cfg: dict, data_cfg: dict, device):
    if not source_ckpt_path.exists():
        print(f"FATAL: --source-ckpt not found: {source_ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(source_ckpt_path, map_location="cpu", weights_only=False)
    if "student" not in ckpt or "predictor" not in ckpt:
        print(f"FATAL: {source_ckpt_path} missing 'student'/'predictor' keys "
              f"(found: {list(ckpt.keys())}). Pass a COMBINED checkpoint.")
        sys.exit(1)

    student, predictor = build_student_predictor(model_cfg, data_cfg)
    student.load_state_dict(ckpt["student"], strict=False)
    predictor.load_state_dict(ckpt["predictor"], strict=False)
    student = student.to(device).eval()
    predictor = predictor.to(device).eval()
    for p in student.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False
    if hasattr(student, "return_hierarchical"):
        student.return_hierarchical = model_cfg["predict_all"] or model_cfg["n_output_distillation"] > 1
    print(f"Loaded FROZEN V-JEPA student + predictor from {source_ckpt_path}")
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
        token=HF_TOKEN or None,
    )


def sample_train_sigma_t(batch_size: int, device) -> torch.Tensor:
    """Logit-normal sigma sampling, matching Cosmos-Predict2.5's own recipe."""
    normal_sample = torch.randn(batch_size, device=device)
    sigma_t = torch.sigmoid(normal_sample)
    return sigma_t.view(batch_size, 1, 1, 1, 1)


def normalize_for_cosmos_vae(pipe, vae_latent: torch.Tensor) -> torch.Tensor:
    latents_mean = pipe.latents_mean.to(vae_latent.device, vae_latent.dtype)
    latents_std = pipe.latents_std.to(vae_latent.device, vae_latent.dtype)
    return (vae_latent - latents_mean) * latents_std


def denormalize_from_cosmos_vae(pipe, vae_latent: torch.Tensor) -> torch.Tensor:
    latents_mean = pipe.latents_mean.to(vae_latent.device, vae_latent.dtype)
    latents_std = pipe.latents_std.to(vae_latent.device, vae_latent.dtype)
    return vae_latent / latents_std + latents_mean


def raw_batch_to_pil_frames(raw_batch: torch.Tensor) -> list:
    video = (raw_batch[0].detach().float().clamp(0, 1) * 255.0).byte()
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    from PIL import Image
    return [Image.fromarray(frame) for frame in video]


def decode_latents_to_pil_frames(pipe, latents: torch.Tensor, dtype: torch.dtype, num_frames: int) -> list:
    latents = denormalize_from_cosmos_vae(pipe, latents)
    with torch.amp.autocast("cuda", dtype=dtype):
        decoded = pipe.vae.decode(latents.to(dtype), return_dict=False)[0]
    pipe._cosmos_infer_num_frames = num_frames
    decoded = pipe._match_num_frames(decoded, pipe._cosmos_infer_num_frames)
    decoded = decoded.float().clamp(-1, 1)
    video = ((decoded[0] + 1.0) * 127.5).clamp(0, 255).byte()
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    from PIL import Image
    return [Image.fromarray(frame) for frame in video]


def export_video(frames: list, output_mp4: Path, fps: int):
    from diffusers.utils import export_to_video
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_mp4), fps=fps)


@torch.no_grad()
def compute_jepa_condition(raw_batch: torch.Tensor, cfg: dict, mask_generators: list,
                           student, predictor, cond_projector, device) -> torch.Tensor:
    model_cfg = cfg["model"]
    mp_cfg = cfg["mixed_precision"]
    jepa_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[mp_cfg["dtype"]]
    enc_batch = normalize_for_jepa(raw_batch.to(device))
    with torch.amp.autocast("cuda", dtype=jepa_dtype, enabled=mp_cfg["enabled"]):
        mg = mask_generators[0]
        m_enc, m_pred = mg(raw_batch.shape[0])
        m_enc, m_pred = m_enc.to(device), m_pred.to(device)
        n_levels = model_cfg["n_output_distillation"]
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
        jepa_latents = select_projector_jepa_latents(jepa_latents, model_cfg)
    return cond_projector(jepa_latents)


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
    condition_mask = torch.zeros(batch_size, 1, num_latent_frames, latent_h, latent_w,
                                 dtype=dtype, device=device)
    padding_mask = torch.zeros(batch_size, 1, latent_h, latent_w, dtype=dtype, device=device)

    pipe.scheduler.set_timesteps(num_steps, device=device)
    for i, timestep in enumerate(pipe.scheduler.timesteps):
        sigma = pipe.scheduler.sigmas[i].expand(batch_size).to(device=device, dtype=torch.float32)
        with torch.amp.autocast("cuda", dtype=dtype):
            velocity = pipe.transformer(
                hidden_states=x,
                condition_mask=condition_mask,
                padding_mask=padding_mask,
                timestep=sigma,
                encoder_hidden_states=cond_embeds.to(dtype),
                return_dict=False,
            )[0]
        x = pipe.scheduler.step(velocity, timestep, x, return_dict=False)[0]
    return x


@torch.no_grad()
def run_visual_validation(step_val: int, validation_raw: torch.Tensor, validation_key: str,
                          cfg: dict, pipe, dit, student, predictor, cond_projector,
                          mask_generators: list, dtype: torch.dtype, device, output_dir: Path,
                          cosmos_cfg: dict):
    validation_cfg = cosmos_cfg["validation"]
    was_training = dit.training
    projector_was_training = cond_projector.training
    dit.eval()
    cond_projector.eval()
    try:
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.manual_seed(validation_cfg["seed"])
            torch.cuda.manual_seed_all(validation_cfg["seed"])
            val_raw = validation_raw.to(device)
            cond_embeds = compute_jepa_condition(
                val_raw, cfg, mask_generators, student, predictor, cond_projector, device)
            with torch.amp.autocast("cuda", dtype=dtype):
                target_latent = pipe.vae.encode((val_raw * 2.0 - 1.0).to(dtype)).latent_dist.sample()
                target_latent = normalize_for_cosmos_vae(pipe, target_latent)
            latents = sample_video_latents(
                pipe=pipe,
                cond_embeds=cond_embeds,
                latent_shape=target_latent.shape,
                num_steps=validation_cfg["num_inference_steps"],
                dtype=dtype,
                device=device,
                seed=validation_cfg["seed"],
            )
        frames = decode_latents_to_pil_frames(pipe, latents, dtype, cfg["data"]["num_frames"])
        val_dir = output_dir / "validation"
        video_path = val_dir / f"step_{step_val:07d}.mp4"
        export_video(frames, video_path, validation_cfg["fps"])
        meta = {
            "step": step_val,
            "clip_key": validation_key,
            "seed": validation_cfg["seed"],
            "num_inference_steps": validation_cfg["num_inference_steps"],
            "fps": validation_cfg["fps"],
            "latent_shape": list(target_latent.shape),
            "jepa_condition_shape": list(cond_embeds.shape),
        }
        video_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"\n[validation] saved generated video: {video_path}")
    finally:
        if was_training:
            dit.train()
        if projector_was_training:
            cond_projector.train()


def train(cfg: dict, cosmos_cfg: dict, args):
    check_gpu()
    print_cgroup_header(prefix="[m14-cosmos]")
    start_oom_watchdog(prefix="[m14-cosmos]-oom-watchdog")
    device = torch.device("cuda")

    torch.manual_seed(cfg["data"]["seed"])
    np.random.seed(cfg["data"]["seed"])

    output_dir = Path(args.output_dir)
    wipe_output_dir(output_dir, args.cache_policy, label=f"output_dir ({output_dir.name})")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    source_ckpt = resolve_source_ckpt(args)
    student, predictor = load_frozen_jepa(Path(source_ckpt), model_cfg, data_cfg, device)
    mask_generators = build_mask_generators(cfg)
    print(f"Mask generators: {len(mask_generators)}")

    dtype = torch.bfloat16
    pipe, dit, vae, text_encoder = load_cosmos_pipeline(
        cosmos_cfg["model_id"], cosmos_cfg["revision"], dtype, device)
    print(f"Loaded Cosmos pipeline: {cosmos_cfg['model_id']} ({cosmos_cfg['revision']})")
    bypass_crossattn_proj = bool(cosmos_cfg["bypass_cosmos_crossattn_proj"])

    dit = add_lora_to_dit(dit, cosmos_cfg["lora_rank"], cosmos_cfg["lora_alpha"], cosmos_cfg["use_dora"])
    dit.train()

    cosmos_context_dim = infer_cosmos_context_dim(
        dit, cosmos_cfg["cosmos_cross_attn_dim_fallback"], bypass_crossattn_proj)
    context_tokens = int(cosmos_cfg["context_tokens"] if bypass_crossattn_proj else 1)
    configure_cosmos_conditioning(dit, bypass_crossattn_proj)
    print(f"Cosmos conditioning context: {context_tokens} tokens x {cosmos_context_dim} dim")
    cond_projector = CondProjector(
        jepa_dim=model_cfg["embed_dim"],
        cosmos_dim=cosmos_context_dim,
        hidden_mult=cosmos_cfg["projector_hidden_mult"],
        num_context_tokens=context_tokens,
    ).to(device=device, dtype=torch.float32)

    trainable_params = [p for p in dit.parameters() if p.requires_grad] + list(cond_projector.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable / 1e6:.1f}M (LoRA + cond projector)")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=cosmos_cfg["learning_rate"],
        weight_decay=cosmos_cfg["weight_decay"])

    from diffusers.optimization import get_linear_schedule_with_warmup
    total_steps = cosmos_cfg["num_training_steps"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cosmos_cfg["scheduler_warmup_steps"],
        num_training_steps=total_steps,
    )

    ckpt_path = output_dir / f"{CHECKPOINT_PREFIX}{artifact('ckpt_latest_suffix')}"
    start_step = 0
    if ckpt_path.exists():
        resume = torch.load(ckpt_path, map_location=device, weights_only=False)
        dit.load_state_dict(resume["dit_lora"], strict=False)
        cond_projector.load_state_dict(resume["cond_projector"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_step = resume["step"]
        print(f"Resumed from step {start_step}")

    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()
    subset_keys = load_subset(args.subset) if args.subset else set()
    prod = threading.Thread(
        target=decoder_producer_thread,
        args=(cfg, q, stop_event, subset_keys, args.local_data, args.max_tar_files),
        daemon=True)
    prod.start()

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb_run = init_wandb("m14_cosmos_latent_decoder", mode, config={**cfg, **cosmos_cfg},
                         enabled=not args.no_wandb)

    jsonl_path = output_dir / "loss_log.jsonl"
    jsonl_file = open(jsonl_path, "a")

    def _log_step(record: dict):
        jsonl_file.write(json.dumps(record) + "\n")
        jsonl_file.flush()
        os.fsync(jsonl_file.fileno())

    def _save_ckpt(step_val: int):
        atomic_torch_save(ckpt_path, {
            "dit_lora": {k: v for k, v in dit.state_dict().items() if "lora" in k.lower()},
            "cond_projector": cond_projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step_val,
        })

    pbar = make_pbar(total=total_steps, initial=start_step, desc="m14_cosmos_latent_decoder", unit="step")
    step = start_step
    log_loss = 0.0
    validation_raw = None
    validation_key = None
    validation_cfg = cosmos_cfg["validation"]

    print(f"\n=== Finetuning Cosmos-Predict2.5 DiT on frozen V-JEPA latents "
          f"(dataset: {WALKINDIA_DATASET_ID}): {start_step} -> {total_steps} steps ===")
    print(f"LoRA rank={cosmos_cfg['lora_rank']} alpha={cosmos_cfg['lora_alpha']} "
          f"use_dora={cosmos_cfg['use_dora']}")
    print(f"NOTE: text_encoder is loaded but UNUSED for conditioning -- cross-attn "
          f"keys/values come entirely from cond_projector(V-JEPA latents).")
    if validation_cfg["enabled"]:
        print(f"Visual validation: every {validation_cfg['every_n_steps']} steps -> "
              f"{output_dir / 'validation'}")

    try:
        for step in range(start_step, total_steps):
            try:
                msg_type, raw_batch, batch_keys = q.get(timeout=600)
            except queue.Empty:
                print(f"Producer timeout at step {step}/{total_steps}. Saving + stopping.")
                break
            if msg_type == "error":
                print(f"FATAL: producer stream failed at step {step}.")
                sys.exit(1)
            if msg_type == "done":
                print(f"\nData exhausted at step {step}/{total_steps}.")
                break

            raw_batch = raw_batch.to(device)
            actual_bs = raw_batch.shape[0]
            enc_batch = normalize_for_jepa(raw_batch)
            if validation_cfg["enabled"] and validation_raw is None:
                validation_raw = raw_batch[:1].detach().cpu()
                validation_key = batch_keys[0] if batch_keys else "unknown"
                val_dir = output_dir / "validation"
                ref_path = val_dir / "reference_input.mp4"
                export_video(raw_batch_to_pil_frames(validation_raw), ref_path, validation_cfg["fps"])
                (val_dir / "reference_input.json").write_text(json.dumps({
                    "clip_key": validation_key,
                    "source": "first successfully decoded training batch",
                    "num_frames": cfg["data"]["num_frames"],
                    "crop_size": cfg["data"]["crop_size"],
                    "fps": validation_cfg["fps"],
                }, indent=2) + "\n")
                print(f"\n[validation] fixed reference clip: {validation_key}")
                print(f"[validation] saved reference video: {ref_path}")
                if validation_cfg["run_before_training"] and step == start_step:
                    try:
                        run_visual_validation(
                            step_val=step,
                            validation_raw=validation_raw,
                            validation_key=validation_key,
                            cfg=cfg,
                            pipe=pipe,
                            dit=dit,
                            student=student,
                            predictor=predictor,
                            cond_projector=cond_projector,
                            mask_generators=mask_generators,
                            dtype=dtype,
                            device=device,
                            output_dir=output_dir,
                            cosmos_cfg=cosmos_cfg,
                        )
                        cuda_cleanup()
                    except torch.cuda.OutOfMemoryError:
                        cuda_cleanup()
                        print("\n[validation] OOM during step-0 baseline; training will continue. "
                              "Reduce validation.num_inference_steps or disable validation.")

            with torch.no_grad():
                mp_cfg = cfg["mixed_precision"]
                jepa_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[mp_cfg["dtype"]]
                with torch.amp.autocast("cuda", dtype=jepa_dtype, enabled=mp_cfg["enabled"]):
                    mg = mask_generators[0]
                    m_enc, m_pred = mg(actual_bs)
                    m_enc, m_pred = m_enc.to(device), m_pred.to(device)
                    n_levels = model_cfg["n_output_distillation"]
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
                    jepa_latents = select_projector_jepa_latents(jepa_latents, model_cfg)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    vae_input = raw_batch * 2.0 - 1.0
                    clean_latent = vae.encode(vae_input.to(dtype)).latent_dist.sample()
                    clean_latent = normalize_for_cosmos_vae(pipe, clean_latent)

            B = clean_latent.shape[0]
            _, _, latent_t, latent_h, latent_w = clean_latent.shape
            condition_mask = torch.zeros(B, 1, latent_t, latent_h, latent_w, dtype=dtype, device=device)
            padding_mask = torch.zeros(B, 1, latent_h, latent_w, dtype=dtype, device=device)
            noise = torch.randn_like(clean_latent)
            sigma_t = sample_train_sigma_t(B, device).to(clean_latent.dtype)

            xt = noise * sigma_t + clean_latent * (1 - sigma_t)
            timestep = sigma_t.view(B)

            try:
                optimizer.zero_grad()
                with torch.enable_grad():
                    cond_embeds = cond_projector(jepa_latents.detach())
                    if cond_embeds.ndim != 3 or cond_embeds.shape[0] != B:
                        raise RuntimeError(
                            f"Cosmos conditioning must be [B, context_tokens, context_dim], got {tuple(cond_embeds.shape)}"
                        )
                    with torch.amp.autocast("cuda", dtype=dtype):
                        pred_velocity = dit(
                            hidden_states=xt.detach(),
                            condition_mask=condition_mask,
                            padding_mask=padding_mask,
                            timestep=timestep,
                            encoder_hidden_states=cond_embeds.to(dtype),
                            return_dict=False,
                        )[0]
                    target_velocity = noise - clean_latent
                    loss = F.mse_loss(pred_velocity.float(), target_velocity.float())
                    if not loss.requires_grad:
                        n_lora_trainable = sum(
                            p.numel() for n, p in dit.named_parameters()
                            if p.requires_grad and ("lora" in n.lower() or "dora" in n.lower() or "adapter" in n.lower())
                        )
                        n_projector_trainable = sum(p.numel() for p in cond_projector.parameters() if p.requires_grad)
                        raise RuntimeError(
                            "loss has no grad_fn; trainable params are not connected to the DiT forward "
                            f"(grad_enabled={torch.is_grad_enabled()}, "
                            f"lora_trainable={n_lora_trainable}, projector_trainable={n_projector_trainable}, "
                            f"cond_requires_grad={cond_embeds.requires_grad}, "
                            f"pred_requires_grad={pred_velocity.requires_grad})"
                        )
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, cosmos_cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad()
                cuda_cleanup()
                print(f"  OOM at step {step}. Consider lowering optimization.batch_size or crop_size.")
                continue

            log_loss = loss.item()
            lr_val = scheduler.get_last_lr()[0]

            _log_step({"step": step, "loss": round(log_loss, 6), "lr": lr_val})
            log_metrics(wb_run, {"loss/rectified_flow_v": log_loss, "lr": lr_val}, step=step)
            pbar.set_postfix_str(f"loss={log_loss:.4f} lr={lr_val:.2e}")
            pbar.update(1)

            ckpt_interval = cosmos_cfg["checkpoint_every_n_steps"]
            if (step + 1) % ckpt_interval == 0:
                _save_ckpt(step + 1)

            val_interval = validation_cfg["every_n_steps"]
            if (validation_cfg["enabled"] and validation_raw is not None and
                    val_interval > 0 and (step + 1) % val_interval == 0):
                try:
                    run_visual_validation(
                        step_val=step + 1,
                        validation_raw=validation_raw,
                        validation_key=validation_key,
                        cfg=cfg,
                        pipe=pipe,
                        dit=dit,
                        student=student,
                        predictor=predictor,
                        cond_projector=cond_projector,
                        mask_generators=mask_generators,
                        dtype=dtype,
                        device=device,
                        output_dir=output_dir,
                        cosmos_cfg=cosmos_cfg,
                    )
                    cuda_cleanup()
                except torch.cuda.OutOfMemoryError:
                    cuda_cleanup()
                    print(f"\n[validation] OOM at step {step + 1}; training will continue. "
                          "Reduce validation.num_inference_steps or disable validation.")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint.")
        pbar.close()
        jsonl_file.close()
        stop_event.set()
        _save_ckpt(step + 1)
        sys.exit(0)
    finally:
        pbar.close()
        jsonl_file.close()
        stop_event.set()

    if step + 1 == start_step:
        raise RuntimeError("0 successful training steps. Refusing to export. Check data path / GPU.")

    export_path = output_dir / "cosmos_latent_decoder_lora_and_projector.pt"
    torch.save({
        "dit_lora_state_dict": {k: v for k, v in dit.state_dict().items() if "lora" in k.lower()},
        "cond_projector_state_dict": cond_projector.state_dict(),
        "cosmos_model_id": cosmos_cfg["model_id"],
        "cosmos_revision": cosmos_cfg["revision"],
        "lora_rank": cosmos_cfg["lora_rank"],
        "lora_alpha": cosmos_cfg["lora_alpha"],
        "use_dora": cosmos_cfg["use_dora"],
        "jepa_embed_dim": model_cfg["embed_dim"],
        "cosmos_context_dim": cosmos_context_dim,
        "context_tokens": context_tokens,
        "bypass_cosmos_crossattn_proj": bypass_crossattn_proj,
        "source_ckpt": str(source_ckpt),
        "source_hf_repo": args.source_hf_repo,
        "source_hf_filename": args.source_hf_filename,
        "dataset": WALKINDIA_DATASET_ID,
    }, export_path)
    print(f"Saved: {export_path}")

    finish_wandb(wb_run)
    print("\n=== COSMOS LATENT DECODER FINETUNE COMPLETE ===")
    print(f"Steps: {step + 1} | final loss: {log_loss:.4f}")
    print(f"Export: {export_path}")
    print("\nAt inference: load the base Cosmos2_5_PredictBasePipeline, apply the "
          "saved LoRA state dict to pipe.transformer, load cond_projector, run "
          "V-JEPA latents through cond_projector to get encoder_hidden_states, "
          "then sample with the pipeline's own flow-matching scheduler over "
          "several denoising steps -- a single forward() call returns a velocity "
          "prediction at one noise level, not a finished frame. Decode the final "
          "denoised latent with pipe.vae.decode(...) to get pixels, and remember "
          "the [-1,1] <-> [0,1] convention used above.")


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(
        description="Finetune Cosmos-Predict2.5-2B to decode frozen V-JEPA "
                    "predictor latents into pixels (cross-attn conditioning swap), "
                    f"trained on {WALKINDIA_DATASET_ID} (streamed WebDataset).")
    parser.add_argument("--SANITY", action="store_true")
    parser.add_argument("--POC", action="store_true")
    parser.add_argument("--FULL", action="store_true")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--train-config", type=str, required=True)
    parser.add_argument("--cosmos-config", type=str, required=True,
                         help="YAML with cosmos model_id/revision/lora settings (see sample below)")
    parser.add_argument("--source-ckpt", type=str, default=None,
                         help="Optional local COMBINED checkpoint with BOTH 'student' and 'predictor' keys. "
                              "If omitted, downloads --source-hf-filename from --source-hf-repo.")
    parser.add_argument("--source-hf-repo", type=str, default=DEFAULT_SOURCE_HF_REPO,
                         help="HF model repo containing the FactorJEPA/V-JEPA checkpoint.")
    parser.add_argument("--source-hf-filename", type=str, default=DEFAULT_SOURCE_HF_FILENAME,
                         help="Checkpoint filename inside --source-hf-repo.")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--local-data", type=str, default=None,
                         help="Local WalkIndia shard directory, e.g. data/full_local. "
                              "If omitted/missing, the repo auto-downloads shards from HF.")
    parser.add_argument("--max-tar-files", type=int, default=None,
                         help="Limit training data to the first N local/HF TAR shards. "
                              "Use 2 or 3 for a small real-data smoke run.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Override cfg.optimization.batch_size. Recommended first smoke value: 1.")
    parser.add_argument("--disable-validation", action="store_true",
                         help="Disable periodic generated-video validation during training.")
    parser.add_argument("--validation-every-n-steps", type=int, default=None,
                         help="Override cosmos_config.validation.every_n_steps.")
    parser.add_argument("--validation-num-inference-steps", type=int, default=None,
                         help="Override cosmos_config.validation.num_inference_steps.")
    parser.add_argument("--validation-seed", type=int, default=None,
                         help="Override cosmos_config.validation.seed.")
    parser.add_argument("--hf-token", type=str, default=None,
                         help="Overrides HF_TOKEN env var. Required for gated datasets/models "
                              "if HF_TOKEN is not already set in your environment.")
    add_wandb_args(parser)
    add_gpu_mem_arg(parser)
    add_cache_policy_arg(parser)
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        global HF_TOKEN
        HF_TOKEN = args.hf_token

    if not os.environ.get("HF_TOKEN"):
        print("WARNING: no Hugging Face token set (env HF_TOKEN or --hf-token). "
              "Gated HF assets may fail to download without one.")

    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    cfg = load_merged_config(args.model_config, args.train_config)
    cosmos_cfg = load_config(args.cosmos_config)
    cosmos_cfg.setdefault("validation", {})
    cosmos_cfg["validation"].setdefault("enabled", True)
    cosmos_cfg["validation"].setdefault("run_before_training", True)
    cosmos_cfg["validation"].setdefault("every_n_steps", cosmos_cfg["checkpoint_every_n_steps"])
    cosmos_cfg["validation"].setdefault("num_inference_steps", 24)
    cosmos_cfg["validation"].setdefault("seed", 1)
    cosmos_cfg["validation"].setdefault("fps", 16)
    cosmos_cfg.setdefault("context_tokens", 98)
    cosmos_cfg.setdefault("bypass_cosmos_crossattn_proj", True)
    if args.disable_validation:
        cosmos_cfg["validation"]["enabled"] = False
    if args.validation_every_n_steps is not None:
        cosmos_cfg["validation"]["every_n_steps"] = args.validation_every_n_steps
    if args.validation_num_inference_steps is not None:
        cosmos_cfg["validation"]["num_inference_steps"] = args.validation_num_inference_steps
    if args.validation_seed is not None:
        cosmos_cfg["validation"]["seed"] = args.validation_seed
    if args.batch_size is not None:
        cfg["optimization"]["batch_size"] = args.batch_size
    args.local_data = ensure_local_data(args)

    mode_key = "sanity" if args.SANITY else ("poc" if args.POC else "full")
    cosmos_cfg["num_training_steps"] = cosmos_cfg["num_training_steps_by_mode"][mode_key]

    train(cfg, cosmos_cfg, args)


if __name__ == "__main__":
    import traceback as _traceback
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except SystemExit:
        raise
    except BaseException as _exc:
        print(f"\nFATAL (unhandled m14-cosmos exception): {type(_exc).__name__}: {_exc}", file=sys.stderr)
        _traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
