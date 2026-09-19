"""Create deterministic source-video-disjoint manifests for JEPA pixel decoding.

USAGE
  python -u src/utils/jepa_decode_splits.py \
    --manifest data/full_local/full_local.json \
    --decode-config configs/jepa_pixel_decode.yaml \
    --mode poc \
    --output-dir data/jepa_decode_splits/poc \
    --cache-policy 2

The source-video assignment is made before mode-specific clip subsampling. A source
video can therefore never occur in more than one of train/validation/test.
"""
import argparse
import json
import random
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from utils.cache_policy import add_cache_policy_arg, resolve_cache_policy_interactive, wipe_output_dir
from utils.jepa_pixel_decode import atomic_json, audit_source_disjoint, read_clip_manifest, source_video_id
from utils.training import load_config


def clip_stratum(clip_key: str) -> str:
    parts = Path(clip_key).parts
    if len(parts) < 3:
        raise ValueError(f"clip key cannot be stratified: {clip_key!r}")
    return "/".join(parts[:-2])


def group_source_videos(clip_keys: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for key in clip_keys:
        grouped[source_video_id(key)].append(key)
    groups = []
    for source_id, keys in grouped.items():
        stratum_counts = Counter(clip_stratum(key) for key in keys)
        dominant = sorted(stratum_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        groups.append({"source_video_id": source_id, "stratum": dominant, "clip_keys": sorted(keys)})
    return groups


def assign_source_groups(groups: list[dict], split_cfg: dict) -> dict[str, list[str]]:
    names = ("train", "validation", "test")
    ratios = {name: float(split_cfg["ratios"][name]) for name in names}
    if not abs(sum(ratios.values()) - 1.0) < float(split_cfg["ratio_tolerance"]):
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    total_clips = sum(len(group["clip_keys"]) for group in groups)
    global_targets = {name: ratios[name] * total_clips for name in names}
    global_counts = {name: 0 for name in names}
    stratum_totals = Counter()
    for group in groups:
        stratum_totals[group["stratum"]] += len(group["clip_keys"])
    stratum_counts = defaultdict(lambda: {name: 0 for name in names})
    assignments = {name: [] for name in names}

    shuffled = list(groups)
    random.Random(int(split_cfg["seed"])).shuffle(shuffled)
    shuffled.sort(key=lambda group: len(group["clip_keys"]), reverse=True)
    global_weight = float(split_cfg["global_balance_weight"])
    for group in shuffled:
        stratum = group["stratum"]
        group_size = len(group["clip_keys"])
        scores = {}
        for name in names:
            stratum_target = ratios[name] * stratum_totals[stratum]
            stratum_deficit = (stratum_target - stratum_counts[stratum][name]) / max(stratum_target, 1.0)
            global_deficit = (global_targets[name] - global_counts[name]) / max(global_targets[name], 1.0)
            scores[name] = stratum_deficit + global_weight * global_deficit
        chosen = sorted(names, key=lambda name: (-scores[name], global_counts[name], name))[0]
        assignments[chosen].extend(group["clip_keys"])
        global_counts[chosen] += group_size
        stratum_counts[stratum][chosen] += group_size
    return assignments


def subsample_split(keys: list[str], requested: int, seed: int) -> list[str]:
    if requested < 0:
        raise ValueError(f"requested sample count must be non-negative, got {requested}")
    if requested == 0:
        return sorted(keys)
    if requested > len(keys):
        raise ValueError(f"requested {requested} clips but source-safe partition has only {len(keys)}")
    ordered = sorted(keys)
    random.Random(seed).shuffle(ordered)
    return sorted(ordered[:requested])


def main():
    parser = argparse.ArgumentParser("Create source-video-safe JEPA decoder splits")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--decode-config", required=True)
    parser.add_argument("--mode", required=True, choices=["sanity", "poc", "full"])
    parser.add_argument("--output-dir", required=True)
    add_cache_policy_arg(parser)
    args = parser.parse_args()
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    config = load_config(args.decode_config)
    split_cfg = config["splits"]
    output_dir = Path(args.output_dir)
    wipe_output_dir(output_dir, args.cache_policy, label="JEPA decoder split output")
    output_dir.mkdir(parents=True, exist_ok=True)

    universe = read_clip_manifest(args.manifest)
    groups = group_source_videos(universe)
    assignments = assign_source_groups(groups, split_cfg)
    mode_counts = split_cfg["samples"][args.mode]
    offsets = split_cfg["subsample_seed_offsets"]
    selected = {
        name: subsample_split(
            assignments[name],
            int(mode_counts[name]),
            int(split_cfg["seed"]) + int(offsets[name]),
        )
        for name in ("train", "validation", "test")
    }
    audit = audit_source_disjoint(selected["train"], selected["validation"], selected["test"])
    assignment_audit = audit_source_disjoint(
        assignments["train"], assignments["validation"], assignments["test"])

    for name in ("train", "validation", "test"):
        atomic_json(output_dir / split_cfg["manifest_filenames"][name], {
            "clip_keys": selected[name],
            "mode": args.mode,
            "split": name,
            "source_manifest": str(Path(args.manifest).resolve()),
            "seed": int(split_cfg["seed"]),
        })
    atomic_json(output_dir / split_cfg["audit_filename"], {
        "mode": args.mode,
        "universe_clips": len(universe),
        "universe_source_videos": len(groups),
        "canonical_assignment": assignment_audit,
        "selected": audit,
        "ratios": split_cfg["ratios"],
        "requested_samples": mode_counts,
    })
    print(json.dumps(audit, indent=2))
    print(f"Saved source-video-safe manifests under {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"FATAL (jepa-decode-splits): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
