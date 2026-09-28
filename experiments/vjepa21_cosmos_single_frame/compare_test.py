"""Paired 500-clip test evaluation of four frozen frame-15 readouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from experiments.vjepa21_cosmos_single_frame.adapter import build_adapter
from experiments.vjepa21_cosmos_single_frame.common import (
    atomic_json_dump, checkpoint_fingerprint, load_config, project_path,
    require_cuda, require_secret,
    resize_center_crop_uint8,
)
from experiments.vjepa21_cosmos_single_frame.data import CACHE_SCHEMA
from experiments.vjepa21_cosmos_single_frame.media import save_seven_panel_test_video
from experiments.vjepa21_cosmos_single_frame.models import (
    CosmosContinuousImageTokenizer, validate_model_geometry,
)
from experiments.vjepa21_cosmos_single_frame.tracking import log_file_artifact, start_wandb

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from utils.data_download import iter_clips_parallel
from utils.video_io import decode_video_bytes

NAMES = ("linear", "latent_mse", "image_loss", "vgg_image_loss")
LABELS = (
    "Context: frames 1-14", "GT frame 15", "Cosmos reconstruction",
    "Linear readout", "Latent-MSE readout", "Image-loss readout",
    "VGG image-loss readout",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(config: dict, alias: str, *, allow_hf: bool) -> Path:
    local = project_path(config["experiment"]["output_dir"]) / f"adapter_{alias}.pt"
    if local.is_file():
        return local
    if not allow_hf:
        raise FileNotFoundError(f"Required local linear checkpoint not found: {local}")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=config["huggingface"]["repo_id"], repo_type="model",
        filename=f"checkpoints/{alias}.pt",
        token=require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",)),
    ))


def check_checkpoint(source: dict, base: dict, name: str) -> None:
    for section, fields in (
        ("experiment", ("seed",)),
        ("data", ("cache_root", "local_root", "train_samples", "val_samples",
                  "test_samples", "context_frames", "target_frame_offset", "crop_size")),
        ("vjepa21", ("model_id", "embed_dim", "patch_size", "tubelet_size")),
        ("cosmos", ("model_id", "latent_channels", "spatial_compression")),
    ):
        for field in fields:
            if source.get(section, {}).get(field) != base[section][field]:
                raise ValueError(f"{name} checkpoint mismatch: {section}.{field}")
    if source.get("training", {}).get("input_latent") != "predicted":
        raise ValueError(f"{name} was not trained on predicted JEPA features")
    expected_type = "linear" if name == "linear" else "residual"
    if source.get("adapter", {}).get("type", "residual") != expected_type:
        raise ValueError(f"{name} has the wrong adapter architecture")


def test_index(base: dict, count: int, seed: int):
    root = project_path(base["data"]["cache_root"]) / "test"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = manifest.get("metadata", {})
    if metadata.get("schema") != CACHE_SCHEMA or metadata.get("target_frame_number") != 15:
        raise RuntimeError("Expected the single-frame frame-15 test cache")
    expected_metadata = {
        "cosmos_model": base["cosmos"]["model_id"],
        "vjepa21_checkpoint": checkpoint_fingerprint(
            project_path(base["vjepa21"]["checkpoint_path"])),
        "crop_size": base["data"]["crop_size"],
        "context_frames": base["data"]["context_frames"],
        "input_dim": base["adapter"]["input_dim"],
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise RuntimeError(f"Test cache is incompatible at {field}")
    if not metadata.get("contains_jepa_predicted") or not metadata.get("contains_target_rgb"):
        raise RuntimeError("Test cache lacks required predicted features or RGB targets")
    if int(manifest["samples"]) != int(base["data"]["test_samples"]):
        raise RuntimeError("Test cache incomplete; finish caching before comparison")
    index = {}
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        for offset, key in enumerate(shard["keys"]):
            if key in index:
                raise RuntimeError(f"Duplicate test key: {key}")
            index[key] = (path, offset)
    source = project_path(base["data"]["local_root"]) / "test"
    source_keys = json.loads((source / "test.json").read_text())["clip_keys"]
    if set(index) != set(source_keys) or len(source_keys) != len(index):
        raise RuntimeError("Test cache keys do not match the source test split")
    if not 1 <= count <= len(index):
        raise ValueError(f"sample_count must be 1..{len(index)}")
    selected = random.Random(seed).sample(sorted(index), count)
    return source, manifest_path, index, selected


@torch.no_grad()
def metrics(decoded, target_rgb, predicted_latent, target_latent, perceptual):
    prediction = (decoded.float().clamp(-1, 1) + 1) / 2
    target = target_rgb.float() / 255
    mse = F.mse_loss(prediction, target)
    lpips_value = perceptual(prediction * 2 - 1, target * 2 - 1).mean()
    result = {
        "rgb_mse": float(mse),
        "rgb_psnr": float(-10 * torch.log10(mse.clamp_min(1e-10))),
        "rgb_lpips_alex": float(lpips_value),
    }
    if predicted_latent is not None:
        result["latent_mse"] = float(F.mse_loss(
            predicted_latent.float(), target_latent.float()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    comparison_path = project_path(args.config)
    spec = yaml.safe_load(comparison_path.read_text())
    if set(spec["checkpoints"]) != set(NAMES):
        raise ValueError(f"Expected four named checkpoints: {NAMES}")
    base = load_config(spec["base_config"])
    validate_model_geometry(base)
    device = require_cuda()
    source, test_manifest, index, selected = test_index(
        base, int(spec["sample_count"]), int(spec["seed"]))
    output = project_path(spec["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_info = {}
    adapters = {}
    for name in NAMES:
        entry = spec["checkpoints"][name]
        config = load_config(entry["config"])
        path = resolve_checkpoint(config, entry["alias"], allow_hf=name != "linear")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "adapter" not in checkpoint or "config" not in checkpoint:
            raise ValueError(f"Invalid adapter checkpoint: {path}")
        check_checkpoint(checkpoint["config"], base, name)
        if name == "linear" and int(checkpoint.get("epoch", -1)) + 1 != 5:
            raise ValueError("Linear checkpoint must be the completed five-epoch run")
        adapter = build_adapter(config).to(device).eval()
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
        adapters[name] = adapter
        checkpoint_info[name] = {
            "path": str(path), "sha256": file_hash(path),
            "epoch": checkpoint.get("epoch"),
            "optimizer_step": checkpoint.get("optimizer_step"),
            "hf_repo": config["huggingface"]["repo_id"] if name != "linear" else None,
        }
        del checkpoint
    provenance = {
        "sample_count": len(selected), "seed": int(spec["seed"]),
        "test_manifest_sha256": file_hash(test_manifest),
        "checkpoint_info": checkpoint_info, "selected_keys": selected,
        "labels": LABELS, "lpips_evaluation_backbone": "alex",
        "selection_protocol": "fixed-seed random sample of disjoint test split; no checkpoint selection",
    }
    provenance_path = output / "comparison_manifest.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise RuntimeError(f"Comparison provenance changed; choose a new output_dir: {output}")
    atomic_json_dump(provenance, provenance_path)

    import lpips
    import wandb
    decoder = CosmosContinuousImageTokenizer(base, device, False, True)
    perceptual = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
    precision = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
        base["training"]["mixed_precision"]]
    run = None if args.no_wandb else start_wandb(
        base, "paired-test-comparison", spec["wandb_run_name"])
    if run:
        run.summary["comparison_manifest"] = str(provenance_path)
        run.summary["test_samples"] = len(selected)
        run.summary["checkpoints"] = checkpoint_info
        run.summary["checkpoint_selection_on_test"] = False

    @lru_cache(maxsize=4)
    def shard_payload(path: Path):
        return torch.load(path, map_location="cpu", weights_only=True)

    rows = {}
    queue, stop_event, reader = iter_clips_parallel(
        str(source), subset_keys=set(selected), num_readers=4, max_queue=16)
    selected_offsets = {key: i for i, key in enumerate(selected)}
    try:
        with tempfile.TemporaryDirectory(prefix="vjepa21_compare_500_") as temporary:
            while True:
                item = queue.get(timeout=300)
                if item is None:
                    break
                key, video_bytes = item
                sample_number = selected_offsets[key]
                shard, offset = index[key]
                payload = shard_payload(shard)
                if payload["keys"][offset] != key:
                    raise RuntimeError(f"Cache shard index mismatch: {key}")
                raw = decode_video_bytes(video_bytes, temporary, key,
                                         int(base["data"]["num_frames"]), False)
                if raw is None:
                    raise RuntimeError(f"Could not decode selected test video: {key}")
                cropped = resize_center_crop_uint8(raw, int(base["data"]["crop_size"]))
                target_cpu = payload["target_rgb"][offset]
                if not torch.equal(cropped[14], target_cpu):
                    raise RuntimeError(f"Source video and cached target differ: {key}")
                jepa = payload["jepa_predicted"][offset].unsqueeze(0).to(device)
                target_latent = payload["cosmos_target"][offset].unsqueeze(0).to(device)
                target_rgb = target_cpu.unsqueeze(0).to(device)
                frame_results = {}
                images = {}
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=precision):
                        cosmos_rgb = decoder.decode(target_latent)
                    frame_results["cosmos"] = metrics(cosmos_rgb, target_rgb, None,
                                                       target_latent, perceptual)
                    for name in NAMES:
                        with torch.autocast("cuda", dtype=precision):
                            z = adapters[name](jepa, tuple(target_latent.shape[-2:]))
                            rgb = decoder.decode(z)
                        frame_results[name] = metrics(rgb, target_rgb, z, target_latent,
                                                      perceptual)
                        images[name] = rgb[0].detach().cpu()
                video_path = output / "videos" / f"sample_{sample_number:03d}.mp4"
                save_seven_panel_test_video(
                    video_path, cropped[:14],
                    [target_cpu, cosmos_rgb[0].detach().cpu(),
                     *(images[name] for name in NAMES)],
                    list(LABELS), panel_size=int(spec["panel_size"]), fps=int(spec["fps"]),
                )
                rows[key] = {"sample_index": sample_number, "metrics": frame_results,
                             "video": str(video_path)}
                atomic_json_dump(rows, output / "per_sample_metrics.json")
                if run:
                    run.log({
                        "comparison/sample_index": sample_number,
                        "comparison/clip_key": key,
                        f"comparison/video_{sample_number:03d}": wandb.Video(
                            str(video_path), format="mp4"),
                        **{f"sample/{name}/{metric}": value
                           for name, group in frame_results.items()
                           for metric, value in group.items()},
                    })
                print(f"Compared {len(rows)}/{len(selected)}: {key}", flush=True)
    finally:
        stop_event.set()
        reader.join(timeout=10)

    if set(rows) != set(selected):
        missing = set(selected) - set(rows)
        raise RuntimeError(f"Only compared {len(rows)}/{len(selected)}; missing: {list(missing)[:5]}")
    summary = {
        name: {metric: sum(rows[key]["metrics"][name][metric] for key in selected) / len(selected)
               for metric in rows[selected[0]]["metrics"][name]}
        for name in (*NAMES, "cosmos")
    }
    summary_path = output / "summary_metrics.json"
    atomic_json_dump({"samples": len(selected), "metrics": summary}, summary_path)
    if run:
        run.log({f"test500/{name}/{metric}": value
                 for name, group in summary.items() for metric, value in group.items()})
        log_file_artifact(run, "vjepa21-75k-paired-test-500-results", "results", [
            provenance_path, output / "per_sample_metrics.json", summary_path])
        run.finish()
    print(json.dumps({"samples": len(selected), "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
