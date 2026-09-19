"""Select one deterministic clip from an existing JSON manifest.

USAGE
  python -u src/utils/jepa_one_clip_manifest.py \
    --source-manifest data/full_local/manifest.json \
    --clip-index 0 \
    --output-manifest data/jepa_decode_splits/overfit_one/one.json \
    --cache-policy 2
"""
import argparse
import json
import os
import traceback
from pathlib import Path

from utils.cache_policy import (
    add_cache_policy_arg,
    guarded_delete,
    resolve_cache_policy_interactive,
)


def read_keys(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        keys = payload
    elif isinstance(payload, dict) and "saved_keys" in payload:
        keys = payload["saved_keys"]
    elif isinstance(payload, dict) and "clip_keys" in payload:
        keys = payload["clip_keys"]
    else:
        raise ValueError("source manifest must be a list or contain saved_keys/clip_keys")
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("source manifest contains no valid clip keys")
    if len(set(keys)) != len(keys):
        raise ValueError("source manifest contains duplicate clip keys")
    return keys


def write_atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser("Create a one-clip JEPA decoding manifest")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--clip-index", required=True, type=int)
    parser.add_argument("--output-manifest", required=True)
    add_cache_policy_arg(parser)
    args = parser.parse_args()
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    source_path = Path(args.source_manifest)
    output_path = Path(args.output_manifest)
    keys = read_keys(source_path)
    if not 0 <= args.clip_index < len(keys):
        raise IndexError(
            f"clip-index {args.clip_index} is outside manifest range [0, {len(keys) - 1}]"
        )
    payload = {
        "clip_keys": [keys[args.clip_index]],
        "source_manifest": str(source_path.resolve()),
        "clip_index": args.clip_index,
    }
    if output_path.exists() and args.cache_policy in {"1", "keep"}:
        existing = json.loads(output_path.read_text())
        if existing != payload:
            raise RuntimeError(
                "kept output manifest differs from the requested selection; "
                "rerun with --cache-policy 2"
            )
        print(f"Preserved matching one-clip manifest: {output_path}")
        return
    guarded_delete(output_path, args.cache_policy, label="one-clip manifest")
    write_atomic_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL (jepa-one-clip-manifest): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
