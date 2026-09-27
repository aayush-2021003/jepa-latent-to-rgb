"""Download an exact, source-disjoint subset from DenseWorld's TAR archive.

The current public ``denseworld-115k`` repository is metadata-only. The original
WebDataset files are in the gated ``denseworld-115k_archive`` repository. This
script downloads a small set of evenly spaced remote shards, scans them locally,
builds train/validation splits with no source-video overlap, and repacks only the
selected samples into FactorJEPA's local WebDataset layout.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import tarfile
from dataclasses import dataclass
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from experiments.jepa_cosmos.common import (
    atomic_json_dump,
    clip_key_from_metadata,
    load_config,
    project_path,
    require_secret,
)
from experiments.jepa_cosmos.tracking import log_file_artifact, start_wandb


@dataclass(frozen=True)
class Record:
    remote_file: str
    local_tar: str
    member_base: str
    clip_key: str
    source_group: str
    tour_type: str = ""


class InsufficientClipsError(RuntimeError):
    """The scanned shards cannot yet supply the requested disjoint splits."""


class RollingTarWriter:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_index = 0
        self.in_shard = 0
        self.total = 0
        self.files: list[str] = []
        self.handle: tarfile.TarFile | None = None
        output_dir.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        filename = f"subset-{self.shard_index:05d}.tar"
        self.files.append(filename)
        self.handle = tarfile.open(self.output_dir / filename, "w")
        self.shard_index += 1
        self.in_shard = 0

    def add(self, base: str, mp4_bytes: bytes, json_bytes: bytes) -> None:
        if self.handle is None or self.in_shard >= self.shard_size:
            self.close_current()
            self._open()
        assert self.handle is not None
        for extension, payload in (("mp4", mp4_bytes), ("json", json_bytes)):
            info = tarfile.TarInfo(name=f"{base}.{extension}")
            info.size = len(payload)
            self.handle.addfile(info, io.BytesIO(payload))
        self.in_shard += 1
        self.total += 1

    def close_current(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def close(self) -> None:
        self.close_current()


def choose_spread_shards(files: list[str], count: int, seed: int) -> list[str]:
    if count > len(files):
        raise ValueError(f"Requested {count} shards from an archive containing {len(files)}")
    if count == len(files):
        return files
    stride = len(files) / count
    rng = random.Random(seed)
    indices = []
    for bucket in range(count):
        start = int(math.floor(bucket * stride))
        end = max(start, int(math.floor((bucket + 1) * stride)) - 1)
        indices.append(rng.randint(start, min(end, len(files) - 1)))
    return [files[index] for index in sorted(set(indices))]


def scan_tar(
    remote_file: str, local_tar: Path, allowed_tour_types: set[str] | None = None
) -> list[Record]:
    records = []
    with tarfile.open(local_tar, "r") as archive:
        json_members = [member for member in archive.getmembers() if member.name.endswith(".json")]
        names = {member.name for member in archive.getmembers()}
        for member in json_members:
            base = member.name.rsplit(".", 1)[0]
            if f"{base}.mp4" not in names:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            metadata = json.loads(extracted.read())
            tour_type = str(metadata.get("tour_type", "")).lower()
            if allowed_tour_types is not None and tour_type not in allowed_tour_types:
                continue
            clip_key = clip_key_from_metadata(metadata)
            video_id = str(metadata["video_id"])
            records.append(
                Record(
                    remote_file=remote_file,
                    local_tar=str(local_tar),
                    member_base=base,
                    clip_key=clip_key,
                    source_group=video_id[:11],
                    tour_type=tour_type,
                )
            )
    return records


def select_source_disjoint(
    records: list[Record], train_samples: int, val_samples: int, seed: int
) -> tuple[list[Record], list[Record]]:
    by_source: dict[str, list[Record]] = {}
    for record in records:
        by_source.setdefault(record.source_group, []).append(record)
    groups = sorted(by_source)
    random.Random(seed).shuffle(groups)

    validation_groups: set[str] = set()
    validation_pool: list[Record] = []
    for group in groups:
        validation_groups.add(group)
        validation_pool.extend(by_source[group])
        if len(validation_pool) >= val_samples:
            break

    train_pool = [record for record in records if record.source_group not in validation_groups]
    rng = random.Random(seed + 1)
    rng.shuffle(validation_pool)
    rng.shuffle(train_pool)
    if len(validation_pool) < val_samples or len(train_pool) < train_samples:
        raise InsufficientClipsError(
            "The selected remote shards do not contain enough source-disjoint clips. "
            "Increase data.download_margin_shards and retry."
        )
    validation = validation_pool[:val_samples]
    train = train_pool[:train_samples]
    if {item.source_group for item in train} & {item.source_group for item in validation}:
        raise AssertionError("Source-video leakage detected between train and validation")
    return train, validation


def select_three_way_source_disjoint(
    records: list[Record], train_samples: int, val_samples: int,
    test_samples: int, eval_source_groups: int, seed: int,
) -> dict[str, list[Record]]:
    """Reserve many entire source videos for each evaluation split.

    Evaluation clips are drawn round-robin across reserved videos so that the
    reported metrics are not dominated by a handful of near-duplicate clips.
    """
    if eval_source_groups < 1 or min(train_samples, val_samples, test_samples) < 1:
        raise ValueError("Three-way split counts and eval_source_groups must be positive")
    if min(val_samples, test_samples) < eval_source_groups:
        raise ValueError("Evaluation sample counts must be >= eval_source_groups")
    by_source: dict[str, list[Record]] = {}
    for record in records:
        by_source.setdefault(record.source_group, []).append(record)
    groups = sorted(by_source)
    rng = random.Random(seed)
    rng.shuffle(groups)
    if len(groups) < 2 * eval_source_groups + 1:
        raise InsufficientClipsError("Too few distinct source videos; increase data.download_margin_shards")
    reserved = {
        "val": groups[:eval_source_groups],
        "test": groups[eval_source_groups:2 * eval_source_groups],
    }
    result: dict[str, list[Record]] = {}
    for split, count in (("val", val_samples), ("test", test_samples)):
        queues = []
        for group in reserved[split]:
            clips = by_source[group].copy()
            rng.shuffle(clips)
            queues.append(clips)
        selected: list[Record] = []
        while len(selected) < count and any(queues):
            for queue in queues:
                if queue and len(selected) < count:
                    selected.append(queue.pop())
        if len(selected) != count or len({row.source_group for row in selected}) < eval_source_groups:
            raise InsufficientClipsError(f"Insufficient {split} clips from reserved source videos; download more shards")
        result[split] = selected
    train_groups = set(groups[2 * eval_source_groups:])
    train_pool = [row for row in records if row.source_group in train_groups]
    rng.shuffle(train_pool)
    if len(train_pool) < train_samples:
        raise InsufficientClipsError(
            f"Only {len(train_pool)} train clips remain after source-disjoint eval reservations; "
            "increase data.download_margin_shards"
        )
    result["train"] = train_pool[:train_samples]
    return result


def repack(
    split_records: dict[str, list[Record]],
    roots: dict[str, Path],
    shard_size: int,
) -> dict[str, dict]:
    lookup = {
        (record.local_tar, record.member_base): split
        for split, records in split_records.items()
        for record in records
    }
    writers = {
        split: RollingTarWriter(root / "m00d_download_subset", shard_size)
        for split, root in roots.items()
    }
    local_tars = sorted({record.local_tar for records in split_records.values() for record in records})
    for local_tar in tqdm(local_tars, desc="Repacking selected clips", unit="tar"):
        with tarfile.open(local_tar, "r") as archive:
            members = {member.name: member for member in archive.getmembers()}
            bases = {name.rsplit(".", 1)[0] for name in members if name.endswith(".json")}
            for base in bases:
                split = lookup.get((local_tar, base))
                if split is None:
                    continue
                mp4_member = members.get(f"{base}.mp4")
                json_member = members.get(f"{base}.json")
                if mp4_member is None or json_member is None:
                    raise RuntimeError(f"Incomplete pair in {local_tar}: {base}")
                mp4_handle = archive.extractfile(mp4_member)
                json_handle = archive.extractfile(json_member)
                if mp4_handle is None or json_handle is None:
                    raise RuntimeError(f"Unreadable pair in {local_tar}: {base}")
                writers[split].add(base, mp4_handle.read(), json_handle.read())
    for writer in writers.values():
        writer.close()

    manifests = {}
    for split, records in split_records.items():
        writer = writers[split]
        if writer.total != len(records):
            raise RuntimeError(f"{split}: repacked {writer.total}, expected {len(records)}")
        manifests[split] = {
            "n": len(records),
            "saved_keys": [record.clip_key for record in records],
            "shards": writer.files,
            "source_videos": len({record.source_group for record in records}),
        }
        atomic_json_dump(manifests[split], roots[split] / "manifest.json")
        atomic_json_dump(
            {"clip_keys": manifests[split]["saved_keys"]},
            roots[split] / f"{split}.json",
        )
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--keep-source-shards", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    allowed_tour_types = data_cfg.get("allowed_tour_types")
    if allowed_tour_types is not None:
        if not isinstance(allowed_tour_types, list) or not allowed_tour_types:
            raise ValueError("data.allowed_tour_types must be a non-empty list")
        allowed_tour_types = {str(value).lower() for value in allowed_tour_types}
        if not allowed_tour_types <= {"walking", "drive", "drone", "rain"}:
            raise ValueError(f"Unknown DenseWorld tour types: {sorted(allowed_tour_types)}")
    token = require_secret("HF_TOKEN", aliases=("HF_ACCESS_TOKEN",))

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    try:
        all_files = api.list_repo_files(data_cfg["archive_repo"], repo_type="dataset")
    except Exception as error:
        raise RuntimeError(
            "DenseWorld archive access failed. Accept the dataset terms at "
            "https://huggingface.co/datasets/anonymousML123/denseworld-115k_archive "
            "for the account owning HF_TOKEN, then retry."
        ) from error
    remote_tars = sorted(
        filename
        for filename in all_files
        if filename.startswith(f"{data_cfg['archive_prefix']}/") and filename.endswith(".tar")
    )
    if not remote_tars:
        raise RuntimeError("No TAR shards found in the DenseWorld archive")

    test_samples = int(data_cfg.get("test_samples", 0))
    total_samples = data_cfg["train_samples"] + data_cfg["val_samples"] + test_samples
    required_shards = math.ceil(total_samples / data_cfg["clips_per_remote_shard"])
    download_count = required_shards + data_cfg["download_margin_shards"]
    selected_shards = choose_spread_shards(
        remote_tars, download_count, config["experiment"]["seed"]
    )
    local_root = project_path(data_cfg["local_root"])
    existing_summary = local_root / "dataset_summary.json"
    if existing_summary.is_file():
        prepared = json.loads(existing_summary.read_text())
        split_names = ("train", "val", "test") if test_samples else ("train", "val")
        expected = tuple(int(data_cfg[f"{split}_samples"]) for split in split_names)
        found = tuple(prepared[split]["n"] for split in split_names)
        if found != expected:
            raise RuntimeError(
                f"Existing dataset at {local_root} has train/val={found}, requested={expected}. "
                "Use a new data.local_root to avoid mixing stale shards."
            )
        if sorted(prepared.get("allowed_tour_types", [])) != sorted(allowed_tour_types or []):
            raise RuntimeError(
                f"Existing dataset at {local_root} uses different tour types; "
                "use a new data.local_root."
            )
        print(json.dumps(prepared, indent=2))
        return
    remote_cache = local_root / "_remote_cache"
    remote_cache.mkdir(parents=True, exist_ok=True)

    run = None if args.no_wandb else start_wandb(config, "dataset-preparation")
    records: list[Record] = []
    downloaded_paths: list[Path] = []
    def download_and_scan(remote_file: str) -> None:
        local_file = Path(
            hf_hub_download(
                repo_id=data_cfg["archive_repo"],
                repo_type="dataset",
                filename=remote_file,
                token=token,
                local_dir=str(remote_cache),
            )
        )
        downloaded_paths.append(local_file)
        records.extend(scan_tar(remote_file, local_file, allowed_tour_types))

    for remote_file in tqdm(selected_shards, desc="Downloading DenseWorld shards", unit="tar"):
        download_and_scan(remote_file)

    clip_keys = [record.clip_key for record in records]
    if len(clip_keys) != len(set(clip_keys)):
        raise RuntimeError("Duplicate clip keys were found across the selected DenseWorld shards")

    def select_splits() -> dict[str, list[Record]]:
        if test_samples:
            return select_three_way_source_disjoint(
                records, data_cfg["train_samples"], data_cfg["val_samples"],
                test_samples, int(data_cfg["eval_source_groups"]), config["experiment"]["seed"],
            )
        train, validation = select_source_disjoint(
            records, data_cfg["train_samples"], data_cfg["val_samples"],
            config["experiment"]["seed"],
        )
        return {"train": train, "val": validation}

    try:
        splits = select_splits()
    except InsufficientClipsError:
        already_selected = set(selected_shards)
        remaining_shards = [name for name in remote_tars if name not in already_selected]
        splits = None
        for remote_file in tqdm(remaining_shards, desc="Backfilling DenseWorld shards", unit="tar"):
            download_and_scan(remote_file)
            selected_shards.append(remote_file)
            splits = None
            try:
                splits = select_splits()
            except InsufficientClipsError:
                continue
            break
        if splits is None:
            raise InsufficientClipsError(
                "The full archive cannot supply the requested source-disjoint subset "
                "after tour-type filtering. Reduce split counts or eval_source_groups."
            )
        if len(records) != len({record.clip_key for record in records}):
            raise RuntimeError("Duplicate clip keys were found during shard backfill")
    roots = {split: local_root / split for split in splits}
    manifests = repack(
        splits,
        roots,
        data_cfg["local_shard_size"],
    )
    summary = {
        "archive_repo": data_cfg["archive_repo"],
        "allowed_tour_types": sorted(allowed_tour_types or []),
        "remote_shards_total": len(remote_tars),
        "remote_shards_downloaded": selected_shards,
        "candidates_scanned": len(records),
        "candidate_tour_type_counts": dict(sorted(Counter(row.tour_type for row in records).items())),
        "split_tour_type_counts": {
            split: dict(sorted(Counter(row.tour_type for row in rows).items()))
            for split, rows in splits.items()
        },
        **manifests,
        "source_overlap": 0,
    }
    summary_path = local_root / "dataset_summary.json"
    atomic_json_dump(summary, summary_path)
    if run is not None:
        run.log(
            {
                **{f"dataset/{split}_samples": len(rows) for split, rows in splits.items()},
                "dataset/remote_shards_downloaded": len(selected_shards),
                "dataset/train_source_videos": manifests["train"]["source_videos"],
                **{f"dataset/{split}_source_videos": manifest["source_videos"] for split, manifest in manifests.items()},
                "dataset/source_overlap": 0,
                **{
                    f"dataset/{split}_{tour_type}_samples": count
                    for split, counts in summary["split_tour_type_counts"].items()
                    for tour_type, count in counts.items()
                },
            }
        )
        log_file_artifact(run, "denseworld-subset-manifests", "dataset", [summary_path])
        run.finish()

    if not args.keep_source_shards:
        for path in downloaded_paths:
            path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
