"""Cache predicted V-JEPA tubelets and frame-15 Cosmos-CI targets."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

from experiments.jepa_cosmos.tracking import log_file_artifact, start_wandb
from experiments.vjepa21_cosmos_single_frame.common import (
    atomic_json_dump,
    atomic_torch_save,
    checkpoint_fingerprint,
    imagenet_normalize,
    load_config,
    project_path,
    require_cuda,
    resize_center_crop_uint8,
    seed_everything,
)
from experiments.vjepa21_cosmos_single_frame.data import CACHE_SCHEMA
from experiments.vjepa21_cosmos_single_frame.models import (
    CosmosContinuousImageTokenizer,
    OfficialVJEPA21WorldModel,
    validate_model_geometry,
)


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from utils.data_download import iter_clips_parallel
from utils.video_io import decode_video_bytes


class CacheWriter:
    def __init__(self, root: Path, samples_per_shard: int, metadata: dict) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self.manifest_path = root / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            for key, value in metadata.items():
                if self.manifest.get("metadata", {}).get(key) != value:
                    raise RuntimeError(
                        f"Cache metadata mismatch for {key!r}; use a new cache_root"
                    )
        else:
            self.manifest = {"metadata": metadata, "samples": 0, "shards": []}
        self.processed = {
            key for shard in self.manifest["shards"] for key in shard["keys"]
        }
        if len(self.processed) != self.manifest["samples"]:
            raise RuntimeError("Cache manifest contains duplicate keys")
        missing = [
            row["file"]
            for row in self.manifest["shards"]
            if not (self.root / row["file"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing cache shards: {missing[:3]}")
        self.buffer: list[dict] = []

    def add(self, sample: dict) -> None:
        self.buffer.append(sample)
        if len(self.buffer) >= self.samples_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        filename = f"latents-{len(self.manifest['shards']):05d}.pt"
        payload = {
            "keys": [sample["key"] for sample in self.buffer],
            "jepa_predicted": torch.stack(
                [sample["jepa_predicted"] for sample in self.buffer]
            ),
            "cosmos_target": torch.stack(
                [sample["cosmos_target"] for sample in self.buffer]
            ),
        }
        for optional in ("jepa_target", "target_rgb"):
            if optional in self.buffer[0]:
                payload[optional] = torch.stack(
                    [sample[optional] for sample in self.buffer]
                )
        atomic_torch_save(payload, self.root / filename)
        entry = {"file": filename, "samples": len(self.buffer), "keys": payload["keys"]}
        self.manifest["shards"].append(entry)
        self.manifest["samples"] += len(self.buffer)
        atomic_json_dump(self.manifest, self.manifest_path)
        self.processed.update(payload["keys"])
        self.buffer.clear()


def load_keys(path: Path) -> list[str]:
    keys = json.loads(path.read_text())["clip_keys"]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate keys in {path}")
    return keys


def decode_batch(items, pool, temporary_dir: str, num_frames: int):
    futures = [
        pool.submit(decode_video_bytes, payload, temporary_dir, key, num_frames, False)
        for key, payload in items
    ]
    decoded = []
    for (key, _), future in zip(items, futures):
        frames = future.result()
        if frames is not None:
            decoded.append((key, frames))
    return decoded


def prepare_split(split: str, config: dict, world, cosmos, run) -> None:
    data = config["data"]
    local_root = project_path(data["local_root"]) / split
    keys = load_keys(local_root / f"{split}.json")
    include_validation = split == "val"
    target_index = int(data["context_frames"]) + int(data["target_frame_offset"])
    metadata = {
        "schema": CACHE_SCHEMA,
        "split": split,
        "vjepa21_checkpoint": checkpoint_fingerprint(
            project_path(config["vjepa21"]["checkpoint_path"])
        ),
        "cosmos_model": config["cosmos"]["model_id"],
        "num_frames": data["num_frames"],
        "context_frames": data["context_frames"],
        "predicted_tubelet_frames": [15, 16],
        "target_frame_number": target_index + 1,
        "crop_size": data["crop_size"],
        "input_dim": config["adapter"]["input_dim"],
        "contains_jepa_predicted": True,
        "contains_jepa_target": include_validation,
        "contains_target_rgb": include_validation,
        "cosmos_target_schema": "continuous_image_frame15_v1",
    }
    cache_root = project_path(data["cache_root"]) / split
    writer = CacheWriter(cache_root, data["cache_samples_per_shard"], metadata)
    remaining = set(keys) - writer.processed
    if not remaining:
        print(f"{split}: cache already complete ({len(keys)} samples)")
        return

    queue, stop_event, reader = iter_clips_parallel(
        str(local_root), subset_keys=remaining, num_readers=data["decode_workers"]
    )
    preview_path = cache_root / "preview.pt"
    preview = [] if not preview_path.exists() else torch.load(preview_path, weights_only=False)
    preview_keys = {sample["key"] for sample in preview}
    progress = tqdm(total=len(remaining), desc=f"Caching {split}", unit="clip")
    pending = []
    failures = 0
    temporary_dir = tempfile.mkdtemp(prefix=f"vjepa21_cosmos_ci_{split}_")

    def consume(decoded) -> None:
        for key, raw_frames in decoded:
            cropped = resize_center_crop_uint8(raw_frames, int(data["crop_size"]))
            normalized = imagenet_normalize(cropped).unsqueeze(0)
            context_end = int(data["context_frames"])
            target_rgb = cropped[target_index]
            predicted = world.predict_future(normalized[:, :context_end])[0].cpu().half()
            expected = (
                int(config["adapter"]["input_dim"]),
                1,
                int(data["crop_size"]) // int(config["vjepa21"]["patch_size"]),
                int(data["crop_size"]) // int(config["vjepa21"]["patch_size"]),
            )
            if tuple(predicted.shape) != expected:
                raise RuntimeError(
                    f"Unexpected V-JEPA prediction shape {tuple(predicted.shape)}; expected {expected}"
                )
            cosmos_input = (target_rgb.float() / 127.5 - 1.0).unsqueeze(0).to("cuda")
            cosmos_target = cosmos.encode(cosmos_input)[0].cpu().half()
            expected_cosmos = (
                int(config["cosmos"]["latent_channels"]),
                int(data["crop_size"]) // int(config["cosmos"]["spatial_compression"]),
                int(data["crop_size"]) // int(config["cosmos"]["spatial_compression"]),
            )
            if tuple(cosmos_target.shape) != expected_cosmos:
                raise RuntimeError(
                    f"Unexpected Cosmos-CI shape {tuple(cosmos_target.shape)}; expected {expected_cosmos}"
                )
            sample = {
                "key": key,
                "jepa_predicted": predicted,
                "cosmos_target": cosmos_target,
            }
            if include_validation:
                sample["jepa_target"] = world.target_future(normalized)[0].cpu().half()
                sample["target_rgb"] = target_rgb.cpu()
            writer.add(sample)
            if include_validation and len(preview) < data["preview_samples"] and key not in preview_keys:
                preview.append(sample)
                preview_keys.add(key)
                atomic_torch_save(preview, preview_path)
            progress.update(1)
            if run is not None and progress.n % 25 == 0:
                run.log({f"cache/{split}_samples": writer.manifest["samples"] + len(writer.buffer)})
            if progress.n % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    try:
        with ThreadPoolExecutor(max_workers=int(data["decode_workers"])) as pool:
            while True:
                item = queue.get(timeout=180)
                if item is None:
                    break
                pending.append(item)
                if len(pending) < int(data["decode_workers"]):
                    continue
                decoded = decode_batch(pending, pool, temporary_dir, int(data["num_frames"]))
                failures += len(pending) - len(decoded)
                pending.clear()
                consume(decoded)
            if pending:
                decoded = decode_batch(pending, pool, temporary_dir, int(data["num_frames"]))
                failures += len(pending) - len(decoded)
                consume(decoded)
    finally:
        stop_event.set()
        reader.join(timeout=10)
        writer.flush()
        progress.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)

    if writer.manifest["samples"] != len(keys):
        raise RuntimeError(
            f"{split}: cached {writer.manifest['samples']}/{len(keys)}; failures={failures}"
        )
    if run is not None:
        log_file_artifact(run, f"{split}-vjepa21-cosmos-ci-cache", "metadata", [writer.manifest_path])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_model_geometry(config)
    seed_everything(config["experiment"]["seed"])
    device = require_cuda()
    run = None
    if not args.no_wandb:
        run = start_wandb(
            config, "vjepa21-cosmos-ci-cache", f"{config['tracking']['run_name']}-cache"
        )
    world = OfficialVJEPA21WorldModel(config, device)
    cosmos = CosmosContinuousImageTokenizer(config, device, True, False)
    splits = ("train", "val") if args.split == "all" else (args.split,)
    for split in splits:
        prepare_split(split, config, world, cosmos, run)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
