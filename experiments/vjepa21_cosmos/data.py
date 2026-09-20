"""Streaming shards for the V-JEPA 2.1 predicted-to-Cosmos cache."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


CACHE_SCHEMA = "vjepa21_predicted_to_cosmos_four_frame_v2"


class LatentShardDataset(IterableDataset):
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
            raise FileNotFoundError(f"Latent cache manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("metadata", {}).get("schema") != CACHE_SCHEMA:
            raise RuntimeError(f"Unsupported cache schema in {manifest_path}")
        self.shards = [self.cache_dir / item["file"] for item in self.manifest["shards"]]
        missing = [str(path) for path in self.shards if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing latent cache shards: {missing[:3]}")

    def __len__(self) -> int:
        samples = int(self.manifest["samples"])
        return samples if self.max_samples is None else min(samples, self.max_samples)

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        indices = list(range(len(self.shards)))
        if worker is not None:
            indices = indices[worker.id :: worker.num_workers]
            seed = self.seed + worker.id
        else:
            seed = self.seed
        rng = random.Random(seed)
        if self.shuffle:
            rng.shuffle(indices)
        yielded = 0
        for shard_index in indices:
            payload = torch.load(
                self.shards[shard_index], map_location="cpu", weights_only=False
            )
            sample_indices = list(range(len(payload["keys"])))
            if self.shuffle:
                rng.shuffle(sample_indices)
            for sample_index in sample_indices:
                if self.max_samples is not None and yielded >= self.max_samples:
                    return
                sample = {
                    "key": payload["keys"][sample_index],
                    "jepa_predicted": payload["jepa_predicted"][sample_index],
                    "cosmos_anchor": payload["cosmos_anchor"][sample_index],
                    "cosmos_target": payload["cosmos_target"][sample_index],
                }
                if "jepa_target" in payload:
                    sample["jepa_target"] = payload["jepa_target"][sample_index]
                if "future_rgb" in payload:
                    sample["future_rgb"] = payload["future_rgb"][sample_index]
                yielded += 1
                yield sample


def cache_collate(samples: list[dict]) -> dict:
    result = {
        "keys": [sample["key"] for sample in samples],
        "jepa_predicted": torch.stack([sample["jepa_predicted"] for sample in samples]),
        "cosmos_anchor": torch.stack([sample["cosmos_anchor"] for sample in samples]),
        "cosmos_target": torch.stack([sample["cosmos_target"] for sample in samples]),
    }
    if "jepa_target" in samples[0]:
        result["jepa_target"] = torch.stack([sample["jepa_target"] for sample in samples])
    if "future_rgb" in samples[0]:
        result["future_rgb"] = torch.stack([sample["future_rgb"] for sample in samples])
    return result
