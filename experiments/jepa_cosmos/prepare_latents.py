"""Precompute frozen JEPA targets/predictions and Cosmos target latents."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

from experiments.jepa_cosmos.common import (
    atomic_json_dump,
    atomic_torch_save,
    checkpoint_fingerprint,
    cosmos_normalize,
    imagenet_normalize,
    load_config,
    project_path,
    require_cuda,
    resize_center_crop_uint8,
    seed_everything,
)
from experiments.jepa_cosmos.models import (
    CosmosContinuousTokenizer,
    FactorJEPAWorldModel,
    validate_model_geometry,
)
from experiments.jepa_cosmos.tracking import log_file_artifact, start_wandb

import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.data_download import iter_clips_parallel
from utils.video_io import decode_video_bytes


def load_keys(path: Path) -> list[str]:
    with path.open() as handle:
        payload = json.load(handle)
    keys = payload["clip_keys"]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate clip keys in {path}")
    return keys


class CacheWriter:
    def __init__(self, root: Path, samples_per_shard: int, metadata: dict) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self.manifest_path = root / "manifest.json"
        if self.manifest_path.exists():
            with self.manifest_path.open() as handle:
                self.manifest = json.load(handle)
            for key, value in metadata.items():
                if self.manifest["metadata"].get(key) != value:
                    raise RuntimeError(f"Cache metadata mismatch for {key}; use a new cache root")
        else:
            self.manifest = {"metadata": metadata, "samples": 0, "shards": []}
        missing_shards = [
            shard["file"]
            for shard in self.manifest["shards"]
            if not (self.root / shard["file"]).is_file()
        ]
        if missing_shards:
            raise RuntimeError(
                f"Cache manifest references missing shards: {missing_shards[:3]}"
            )
        self.processed = {
            key
            for shard in self.manifest["shards"]
            for key in shard["keys"]
        }
        if len(self.processed) != self.manifest["samples"]:
            raise RuntimeError("Cache manifest contains duplicate keys or an invalid sample count")
        self.buffer: list[dict] = []

    def add(self, sample: dict) -> None:
        self.buffer.append(sample)
        if len(self.buffer) >= self.samples_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_index = len(self.manifest["shards"])
        filename = f"latents-{shard_index:05d}.pt"
        payload = {
            "keys": [sample["key"] for sample in self.buffer],
            "jepa_target": torch.stack([sample["jepa_target"] for sample in self.buffer]),
            "cosmos_anchor": torch.stack([sample["cosmos_anchor"] for sample in self.buffer]),
            "cosmos_target": torch.stack([sample["cosmos_target"] for sample in self.buffer]),
        }
        if "jepa_predicted" in self.buffer[0]:
            payload["jepa_predicted"] = torch.stack(
                [sample["jepa_predicted"] for sample in self.buffer]
            )
        atomic_torch_save(payload, self.root / filename)
        entry = {"file": filename, "samples": len(self.buffer), "keys": payload["keys"]}
        self.manifest["shards"].append(entry)
        self.manifest["samples"] += len(self.buffer)
        atomic_json_dump(self.manifest, self.manifest_path)
        self.processed.update(payload["keys"])
        self.buffer.clear()


def decode_batch(
    items: list[tuple[str, bytes]],
    pool: ThreadPoolExecutor,
    temporary_dir: str,
    num_frames: int,
) -> list[tuple[str, torch.Tensor]]:
    futures = [
        pool.submit(decode_video_bytes, mp4, temporary_dir, key, num_frames, False)
        for key, mp4 in items
    ]
    decoded = []
    for (key, _), future in zip(items, futures):
        frames = future.result()
        if frames is not None:
            decoded.append((key, frames))
    return decoded


def prepare_split(
    split: str,
    config: dict,
    world_model: FactorJEPAWorldModel,
    cosmos: CosmosContinuousTokenizer,
    run,
) -> None:
    data_cfg = config["data"]
    local_root = project_path(data_cfg["local_root"]) / split
    keys = load_keys(local_root / f"{split}.json")
    factor_path = project_path(config["factorjepa"]["checkpoint_path"])
    include_prediction = split == "val" or bool(
        data_cfg.get("cache_predicted_train", False)
    )
    metadata = {
        "split": split,
        "factorjepa_checkpoint": checkpoint_fingerprint(factor_path),
        "cosmos_model": config["cosmos"]["model_id"],
        "num_frames": data_cfg["num_frames"],
        "context_frames": data_cfg["context_frames"],
        "crop_size": data_cfg["crop_size"],
        "input_dim": config["adapter"]["input_dim"],
        "cosmos_target_schema": "last_context_anchor_plus_future_v1",
    }
    if data_cfg.get("cache_predicted_train", False):
        metadata["contains_jepa_predicted"] = include_prediction
    cache_root = project_path(data_cfg["cache_root"]) / split
    writer = CacheWriter(cache_root, data_cfg["cache_samples_per_shard"], metadata)
    remaining = set(keys) - writer.processed
    if not remaining:
        preview_path = cache_root / "preview.pt"
        if split == "val" and not preview_path.is_file():
            raise RuntimeError(
                "Validation cache is complete but preview.pt is missing. Use a new "
                "data.cache_root (or rebuild the validation cache) before training."
            )
        print(f"{split}: latent cache already complete ({len(keys)} samples)")
        return

    clip_queue, stop_event, reader = iter_clips_parallel(
        str(local_root), subset_keys=remaining, num_readers=data_cfg["decode_workers"]
    )
    preview_path = cache_root / "preview.pt"
    preview = [] if not preview_path.exists() else torch.load(preview_path, weights_only=False)
    preview_keys = {item["key"] for item in preview}
    progress = tqdm(total=len(remaining), desc=f"Caching {split} latents", unit="clip")
    pending: list[tuple[str, bytes]] = []
    failures = 0
    temporary_dir = tempfile.mkdtemp(prefix=f"jepa_cosmos_{split}_")

    def consume(decoded: list[tuple[str, torch.Tensor]]) -> None:
        for key, raw_frames in decoded:
            cropped = resize_center_crop_uint8(raw_frames, data_cfg["crop_size"])
            normalized = imagenet_normalize(cropped).unsqueeze(0)
            anchor_rgb = cropped[
                data_cfg["context_frames"] - 1:data_cfg["context_frames"]
            ]
            future_rgb = cropped[data_cfg["context_frames"]:]
            anchored_rgb = cropped[data_cfg["context_frames"] - 1:]
            anchor_video = cosmos_normalize(anchor_rgb).unsqueeze(0).to("cuda")
            anchored_video = cosmos_normalize(anchored_rgb).unsqueeze(0).to("cuda")
            jepa_target = world_model.target_future(normalized)[0].cpu().half()
            cosmos_anchor = cosmos.encode(anchor_video)[0].cpu().half()
            anchored_latent = cosmos.encode(anchored_video)[0].cpu().half()
            cosmos_target = anchored_latent[:, 1:]
            sample = {
                "key": key,
                "jepa_target": jepa_target,
                "cosmos_anchor": cosmos_anchor,
                "cosmos_target": cosmos_target,
            }
            jepa_predicted = None
            if include_prediction:
                jepa_predicted = world_model.predict_future(
                    normalized[:, :data_cfg["context_frames"]]
                )[0].cpu().half()
                sample["jepa_predicted"] = jepa_predicted
            writer.add(sample)
            if (
                split == "val"
                and len(preview) < data_cfg["preview_samples"]
                and key not in preview_keys
            ):
                preview.append(
                    {
                        "key": key,
                        "context_rgb": cropped[:data_cfg["context_frames"]],
                        "future_rgb": future_rgb,
                        "jepa_target": jepa_target,
                        "jepa_predicted": jepa_predicted,
                        "cosmos_anchor": cosmos_anchor,
                        "cosmos_target": cosmos_target,
                    }
                )
                preview_keys.add(key)
                atomic_torch_save(preview, preview_path)
            progress.update(1)
            if run is not None and progress.n % 25 == 0:
                run.log(
                    {
                        f"cache/{split}_samples": writer.manifest["samples"] + len(writer.buffer),
                        f"cache/{split}_decode_failures": failures,
                    }
                )
            if progress.n % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    try:
        with ThreadPoolExecutor(max_workers=data_cfg["decode_workers"]) as pool:
            while True:
                item = clip_queue.get(timeout=180)
                if item is None:
                    break
                pending.append(item)
                if len(pending) < data_cfg["decode_workers"]:
                    continue
                decoded = decode_batch(
                    pending, pool, temporary_dir, data_cfg["num_frames"]
                )
                failures += len(pending) - len(decoded)
                pending.clear()
                consume(decoded)
            if pending:
                decoded = decode_batch(
                    pending, pool, temporary_dir, data_cfg["num_frames"]
                )
                failures += len(pending) - len(decoded)
                consume(decoded)
    finally:
        stop_event.set()
        reader.join(timeout=10)
        progress.close()
        writer.flush()
        shutil.rmtree(temporary_dir, ignore_errors=True)

    if writer.manifest["samples"] != len(keys):
        raise RuntimeError(
            f"{split}: cached {writer.manifest['samples']}/{len(keys)} samples; "
            f"decode failures={failures}"
        )
    if run is not None:
        log_file_artifact(
            run,
            f"{split}-latent-cache-manifest",
            "metadata",
            [writer.manifest_path],
        )


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
    run = None if args.no_wandb else start_wandb(config, "latent-precompute")
    world_model = FactorJEPAWorldModel(config, device)
    cosmos = CosmosContinuousTokenizer(config, device, load_encoder=True, load_decoder=False)
    splits = ("train", "val") if args.split == "all" else (args.split,)
    for split in splits:
        prepare_split(split, config, world_model, cosmos, run)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
