"""Streaming dataset for cached V-JEPA 2.1 predicted features."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


CACHE_SCHEMA = "vjepa21_predicted_single_tubelet_to_two_frames_v2"


class TwoFrameCacheDataset(IterableDataset):
    def __init__(
        self,
        cache_dir: str | Path,
        shuffle: bool,
        seed: int,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("metadata", {}).get("schema") != CACHE_SCHEMA:
            raise RuntimeError(f"Unsupported cache schema in {manifest_path}")
        self.shards = [self.cache_dir / item["file"] for item in self.manifest["shards"]]
        missing = [str(path) for path in self.shards if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing cache shards: {missing[:3]}")

    def __len__(self) -> int:
        count = int(self.manifest["samples"])
        return count if self.max_samples is None else min(count, self.max_samples)

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        shard_indices = list(range(len(self.shards)))
        if worker is not None:
            shard_indices = shard_indices[worker.id :: worker.num_workers]
            seed = self.seed + worker.id
        else:
            seed = self.seed
        rng = random.Random(seed)
        if self.shuffle:
            rng.shuffle(shard_indices)

        yielded = 0
        for shard_index in shard_indices:
            payload = torch.load(
                self.shards[shard_index], map_location="cpu", weights_only=False
            )
            indices = list(range(len(payload["keys"])))
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                if self.max_samples is not None and yielded >= self.max_samples:
                    return
                sample = {
                    "key": payload["keys"][index],
                    "predicted_features": payload["predicted_features"][index],
                    "target_rgb": payload["target_rgb"][index],
                    "last_context_rgb": payload["last_context_rgb"][index],
                }
                if "target_features" in payload:
                    sample["target_features"] = payload["target_features"][index]
                yielded += 1
                yield sample


def cache_collate(samples: list[dict]) -> dict:
    result = {
        "keys": [sample["key"] for sample in samples],
        "predicted_features": torch.stack(
            [sample["predicted_features"] for sample in samples]
        ),
        "target_rgb": torch.stack([sample["target_rgb"] for sample in samples]),
        "last_context_rgb": torch.stack(
            [sample["last_context_rgb"] for sample in samples]
        ),
    }
    if "target_features" in samples[0]:
        result["target_features"] = torch.stack(
            [sample["target_features"] for sample in samples]
        )
    return result
