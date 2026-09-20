"""Latent-cache datasets that load one shard at a time."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


class LatentShardDataset(IterableDataset):
    """Stream cached latent shards without repeatedly loading giant feature files."""

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
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.max_samples = max_samples
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Latent cache manifest not found: {manifest_path}")
        with manifest_path.open() as handle:
            self.manifest = json.load(handle)
        if (
            self.manifest.get("metadata", {}).get("cosmos_target_schema")
            != "last_context_anchor_plus_future_v1"
        ):
            raise RuntimeError(
                "Latent cache does not use the anchored Cosmos target schema; rebuild it."
            )
        self.shards = [self.cache_dir / item["file"] for item in self.manifest["shards"]]
        missing = [str(path) for path in self.shards if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing latent cache shards: {missing[:3]}")

    def __len__(self) -> int:
        samples = int(self.manifest["samples"])
        return samples if self.max_samples is None else min(samples, self.max_samples)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | str]]:
        worker = torch.utils.data.get_worker_info()
        shard_indices = list(range(len(self.shards)))
        if worker is not None:
            shard_indices = shard_indices[worker.id::worker.num_workers]
            worker_seed = self.seed + worker.id
        else:
            worker_seed = self.seed
        rng = random.Random(worker_seed)
        if self.shuffle:
            rng.shuffle(shard_indices)

        yielded = 0
        for shard_index in shard_indices:
            payload = torch.load(self.shards[shard_index], map_location="cpu", weights_only=False)
            sample_indices = list(range(len(payload["keys"])))
            if self.shuffle:
                rng.shuffle(sample_indices)
            for sample_index in sample_indices:
                if self.max_samples is not None and yielded >= self.max_samples:
                    return
                sample = {
                    "key": payload["keys"][sample_index],
                    "jepa_target": payload["jepa_target"][sample_index],
                    "cosmos_anchor": payload["cosmos_anchor"][sample_index],
                    "cosmos_target": payload["cosmos_target"][sample_index],
                }
                if "jepa_predicted" in payload:
                    sample["jepa_predicted"] = payload["jepa_predicted"][sample_index]
                yielded += 1
                yield sample


def cache_collate(samples: list[dict]) -> dict:
    result = {
        "keys": [sample["key"] for sample in samples],
        "jepa_target": torch.stack([sample["jepa_target"] for sample in samples]),
        "cosmos_anchor": torch.stack([sample["cosmos_anchor"] for sample in samples]),
        "cosmos_target": torch.stack([sample["cosmos_target"] for sample in samples]),
    }
    if "jepa_predicted" in samples[0]:
        result["jepa_predicted"] = torch.stack(
            [sample["jepa_predicted"] for sample in samples]
        )
    return result
