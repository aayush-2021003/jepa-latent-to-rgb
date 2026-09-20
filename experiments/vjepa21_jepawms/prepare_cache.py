"""Cache official V-JEPA 2.1 predictions and both held-out RGB frames."""
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
import torch.nn.functional as F
from tqdm import tqdm

from experiments.vjepa21_jepawms.common import (
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
from experiments.vjepa21_jepawms.data import CACHE_SCHEMA
from experiments.vjepa21_jepawms.models import (
    OfficialVJEPA21WorldModel,
    validate_geometry,
)
from experiments.jepa_cosmos.tracking import log_file_artifact, start_wandb


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
        for shard in self.manifest["shards"]:
            if not (self.root / shard["file"]).is_file():
                raise RuntimeError(f"Manifest references missing shard {shard['file']}")
        self.buffer: list[dict] = []

    def add(self, sample: dict) -> None:
        self.buffer.append(sample)
        if len(self.buffer) >= self.samples_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        filename = f"features-{len(self.manifest['shards']):05d}.pt"
        payload = {
            "keys": [sample["key"] for sample in self.buffer],
            "predicted_features": torch.stack(
                [sample["predicted_features"] for sample in self.buffer]
            ),
            "target_rgb": torch.stack([sample["target_rgb"] for sample in self.buffer]),
            "last_context_rgb": torch.stack(
                [sample["last_context_rgb"] for sample in self.buffer]
            ),
        }
        if "target_features" in self.buffer[0]:
            payload["target_features"] = torch.stack(
                [sample["target_features"] for sample in self.buffer]
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


def resize_uint8(frame: torch.Tensor, size: int) -> torch.Tensor:
    resized = F.interpolate(
        frame.unsqueeze(0).float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    return resized.round().clamp(0, 255).to(torch.uint8)


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


def prepare_split(split: str, config: dict, model, run) -> None:
    data = config["data"]
    local_root = project_path(data["local_root"]) / split
    keys = load_keys(local_root / f"{split}.json")
    include_target_features = split == "val"
    metadata = {
        "schema": CACHE_SCHEMA,
        "split": split,
        "vjepa21_checkpoint": checkpoint_fingerprint(
            project_path(config["vjepa21"]["checkpoint_path"])
        ),
        "num_frames": data["num_frames"],
        "context_frames": data["context_frames"],
        "target_frame_index": data["target_frame_index"],
        "target_frame_indices": data["target_frame_indices"],
        "crop_size": data["crop_size"],
        "decoder_image_size": data["decoder_image_size"],
        "feature_dim": config["adapter"]["input_dim"],
        "contains_target_features": include_target_features,
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
    progress = tqdm(total=len(remaining), desc=f"Caching {split}", unit="clip")
    pending = []
    failures = 0
    temporary_dir = tempfile.mkdtemp(prefix=f"vjepa21_two_frame_{split}_")

    def consume(decoded) -> None:
        nonlocal failures
        for key, raw_frames in decoded:
            cropped = resize_center_crop_uint8(raw_frames, data["crop_size"])
            normalized = imagenet_normalize(cropped).unsqueeze(0)
            predicted = model.predict_tubelet(
                normalized[:, : data["context_frames"]]
            )[0].cpu().half()
            target_rgb = torch.stack(
                [
                    resize_uint8(cropped[int(index)], data["decoder_image_size"])
                    for index in data["target_frame_indices"]
                ]
            )
            context_rgb = resize_uint8(
                cropped[data["context_frames"] - 1], data["decoder_image_size"]
            )
            sample = {
                "key": key,
                "predicted_features": predicted,
                "target_rgb": target_rgb,
                "last_context_rgb": context_rgb,
            }
            if include_target_features:
                sample["target_features"] = (
                    model.target_tubelet(normalized)[0].cpu().half()
                )
            writer.add(sample)
            progress.update(1)
            if run is not None and progress.n % 25 == 0:
                run.log(
                    {
                        f"cache/{split}_samples": writer.manifest["samples"]
                        + len(writer.buffer),
                        f"cache/{split}_decode_failures": failures,
                    }
                )
            if progress.n % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    try:
        with ThreadPoolExecutor(max_workers=data["decode_workers"]) as pool:
            while True:
                item = queue.get(timeout=180)
                if item is None:
                    break
                pending.append(item)
                if len(pending) < data["decode_workers"]:
                    continue
                decoded = decode_batch(pending, pool, temporary_dir, data["num_frames"])
                failures += len(pending) - len(decoded)
                pending.clear()
                consume(decoded)
            if pending:
                decoded = decode_batch(pending, pool, temporary_dir, data["num_frames"])
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
        log_file_artifact(
            run, f"{split}-vjepa21-cache-manifest", "metadata", [writer.manifest_path]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_geometry(config)
    seed_everything(config["experiment"]["seed"])
    device = require_cuda()
    run = None
    if not args.no_wandb:
        run = start_wandb(
            config,
            "vjepa21-feature-precompute",
            f"{config['tracking']['run_name']}-cache",
        )
    model = OfficialVJEPA21WorldModel(config, device)
    splits = ("train", "val") if args.split == "all" else (args.split,)
    for split in splits:
        prepare_split(split, config, model, run)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
