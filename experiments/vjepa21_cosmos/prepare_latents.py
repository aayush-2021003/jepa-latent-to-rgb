"""Cache predicted V-JEPA 2.1 features and four-frame Cosmos targets."""
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

from experiments.vjepa21_cosmos.common import (
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
from experiments.vjepa21_cosmos.data import CACHE_SCHEMA
from experiments.vjepa21_cosmos.models import (
    CosmosContinuousTokenizer,
    OfficialVJEPA21WorldModel,
    validate_model_geometry,
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
        missing = [
            item["file"]
            for item in self.manifest["shards"]
            if not (self.root / item["file"]).is_file()
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
            "cosmos_anchor": torch.stack(
                [sample["cosmos_anchor"] for sample in self.buffer]
            ),
            "cosmos_target": torch.stack(
                [sample["cosmos_target"] for sample in self.buffer]
            ),
        }
        if "jepa_target" in self.buffer[0]:
            payload["jepa_target"] = torch.stack(
                [sample["jepa_target"] for sample in self.buffer]
            )
        if "future_rgb" in self.buffer[0]:
            payload["future_rgb"] = torch.stack(
                [sample["future_rgb"] for sample in self.buffer]
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
    include_target = split == "val"
    metadata = {
        "schema": CACHE_SCHEMA,
        "split": split,
        "vjepa21_checkpoint": checkpoint_fingerprint(
            project_path(config["vjepa21"]["checkpoint_path"])
        ),
        "cosmos_model": config["cosmos"]["model_id"],
        "num_frames": data["num_frames"],
        "context_frames": data["context_frames"],
        "future_frames": data["num_frames"] - data["context_frames"],
        "crop_size": data["crop_size"],
        "input_dim": config["adapter"]["input_dim"],
        "contains_jepa_predicted": True,
        "contains_jepa_target": include_target,
        "contains_future_rgb": include_target,
        "cosmos_target_schema": "last_context_anchor_plus_four_future_v1",
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
    temporary_dir = tempfile.mkdtemp(prefix=f"vjepa21_cosmos_{split}_")

    def consume(decoded) -> None:
        for key, raw_frames in decoded:
            cropped = resize_center_crop_uint8(raw_frames, data["crop_size"])
            normalized = imagenet_normalize(cropped).unsqueeze(0)
            context_end = data["context_frames"]
            anchor_rgb = cropped[context_end - 1 : context_end]
            future_rgb = cropped[context_end:]
            anchored_rgb = cropped[context_end - 1 :]
            predicted = world.predict_future(normalized[:, :context_end])[0].cpu().half()
            expected_jepa = (
                config["adapter"]["input_dim"],
                2,
                data["crop_size"] // config["vjepa21"]["patch_size"],
                data["crop_size"] // config["vjepa21"]["patch_size"],
            )
            if tuple(predicted.shape) != expected_jepa:
                raise RuntimeError(
                    f"Unexpected V-JEPA prediction shape {tuple(predicted.shape)}; "
                    f"expected {expected_jepa}"
                )
            cosmos_anchor = cosmos.encode(
                cosmos_normalize(anchor_rgb).unsqueeze(0).to("cuda")
            )[0].cpu().half()
            anchored_latent = cosmos.encode(
                cosmos_normalize(anchored_rgb).unsqueeze(0).to("cuda")
            )[0].cpu().half()
            if cosmos_anchor.shape[1] != 1 or anchored_latent.shape[1] != 2:
                raise RuntimeError(
                    "Unexpected Cosmos temporal geometry: "
                    f"anchor={tuple(cosmos_anchor.shape)}, "
                    f"anchored={tuple(anchored_latent.shape)}"
                )
            sample = {
                "key": key,
                "jepa_predicted": predicted,
                "cosmos_anchor": cosmos_anchor,
                "cosmos_target": anchored_latent[:, 1:],
            }
            if include_target:
                sample["jepa_target"] = world.target_future(normalized)[0].cpu().half()
                sample["future_rgb"] = future_rgb.cpu()
            writer.add(sample)
            if split == "val" and len(preview) < data["preview_samples"] and key not in preview_keys:
                preview.append(
                    {
                        **sample,
                        "context_rgb": cropped[:context_end],
                    }
                )
                preview_keys.add(key)
                atomic_torch_save(preview, preview_path)
            progress.update(1)
            if run is not None and progress.n % 25 == 0:
                run.log({f"cache/{split}_samples": writer.manifest["samples"] + len(writer.buffer)})
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
            run, f"{split}-vjepa21-cosmos-cache", "metadata", [writer.manifest_path]
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
    run = None
    if not args.no_wandb:
        run = start_wandb(
            config, "vjepa21-cosmos-cache", f"{config['tracking']['run_name']}-cache"
        )
    world = OfficialVJEPA21WorldModel(config, device)
    cosmos = CosmosContinuousTokenizer(config, device, load_encoder=True, load_decoder=False)
    splits = ("train", "val") if args.split == "all" else (args.split,)
    for split in splits:
        prepare_split(split, config, world, cosmos, run)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
