import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_ENDPOINT"] = os.environ.get("FACTORJEPA_HF_ENDPOINT", "https://huggingface.co")
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = os.environ["HF_ENDPOINT"]

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import stage1_jepa_decoder_train as stage1_train
from stage1_jepa_decoder_train import (
    decode_latents_to_frames,
    encode_video_latents,
    export_video,
    load_cosmos_vae,
)


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Cosmos VAE roundtrip")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def decode_mp4_with_pyav(path: Path, num_frames: int) -> torch.Tensor:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for MP4 decode. Install with: pip install av") from exc

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total = stream.frames or 0
        wanted = None
        if total and total >= num_frames:
            wanted = set(torch.linspace(0, total - 1, steps=num_frames).round().long().tolist())
        for idx, frame in enumerate(container.decode(stream)):
            if wanted is None or idx in wanted:
                arr = frame.to_rgb().to_ndarray()
                frames.append(torch.from_numpy(arr).permute(2, 0, 1))
                if len(frames) >= num_frames:
                    break
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    return torch.stack(frames[:num_frames], dim=0)


def resize_center_crop(video_tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    video = video_tensor.float() / 255.0
    _, _, h, w = video.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    video = video[:, :, top:top + side, left:left + side]
    video = F.interpolate(video, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    return video.permute(1, 0, 2, 3).contiguous()


def load_mp4_as_raw_batch(path: Path, num_frames: int, crop_size: int) -> torch.Tensor:
    decoded = decode_mp4_with_pyav(path, num_frames)
    raw_clip = resize_center_crop(decoded, crop_size)
    return raw_clip.unsqueeze(0)


def raw_batch_to_frames(raw_batch: torch.Tensor) -> list:
    video = (raw_batch[0].detach().float().clamp(0, 1) * 255.0).byte()
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    from PIL import Image
    return [Image.fromarray(frame) for frame in video]


def frames_to_tensor(frames: list) -> torch.Tensor:
    import numpy as np

    arr = np.stack([np.asarray(frame, dtype=np.float32) for frame in frames], axis=0)
    arr = torch.from_numpy(arr).permute(3, 0, 1, 2) / 255.0
    return arr


def compute_pixel_metrics(reference_raw: torch.Tensor, recon_frames: list) -> dict:
    reference = reference_raw[0].detach().cpu().float().clamp(0, 1)
    recon = frames_to_tensor(recon_frames).float().clamp(0, 1)
    if recon.shape != reference.shape:
        raise RuntimeError(f"reconstruction {tuple(recon.shape)} != reference {tuple(reference.shape)}")
    mse = F.mse_loss(recon, reference).item()
    mae = F.l1_loss(recon, reference).item()
    psnr = float("inf") if mse == 0 else -10.0 * torch.log10(torch.tensor(mse)).item()
    return {
        "pixel_mse_0_1": mse,
        "pixel_mae_0_1": mae,
        "pixel_psnr_db": psnr,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Check Cosmos VAE encode/decode roundtrip for one MP4")
    parser.add_argument("--input-mp4", required=True)
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--reference-mp4", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--model-id", default="nvidia/Cosmos-Predict2.5-2B")
    parser.add_argument("--revision", default="diffusers/base/post-trained")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        stage1_train.HF_TOKEN = args.hf_token

    check_gpu()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    input_path = Path(args.input_mp4)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw_batch = load_mp4_as_raw_batch(input_path, args.num_frames, args.crop_size).to(device)
    pipe = load_cosmos_vae(
        args.model_id, args.revision, dtype, device, vae_subfolder="vae")

    latents = encode_video_latents(pipe, raw_batch, dtype)
    recon_frames = decode_latents_to_frames(pipe, latents, dtype, args.num_frames)

    output_path = Path(args.output_mp4)
    export_video(recon_frames, output_path, args.fps)

    reference_path = Path(args.reference_mp4) if args.reference_mp4 else output_path.with_name(
        output_path.stem + "_reference.mp4"
    )
    export_video(raw_batch_to_frames(raw_batch.detach().cpu()), reference_path, args.fps)

    metrics = {
        "input_mp4": str(input_path),
        "output_mp4": str(output_path),
        "reference_mp4": str(reference_path),
        "model_id": args.model_id,
        "revision": args.revision,
        "num_frames": args.num_frames,
        "crop_size": args.crop_size,
        "fps": args.fps,
        "raw_batch_shape": list(raw_batch.shape),
        "latent_shape": list(latents.shape),
    }
    metrics.update(compute_pixel_metrics(raw_batch.detach().cpu(), recon_frames))

    metrics_path = Path(args.metrics_json) if args.metrics_json else output_path.with_suffix(".json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Saved Cosmos VAE reconstruction: {output_path}")
    print(f"Saved reference crop: {reference_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        print(f"\nFATAL (cosmos-vae-roundtrip): {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
