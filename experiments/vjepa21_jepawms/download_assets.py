"""Download the official V-JEPA 2.1 and JEPA-WMs decoder assets."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm

from experiments.vjepa21_jepawms.common import load_config, project_path


def download_url(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
            partial.unlink(missing_ok=True)
        total = int(response.headers.get("content-length", 0)) + offset
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle, tqdm(
            total=total,
            initial=offset,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    os.replace(partial, destination)


def ensure_dependency(repo: str, revision: str, destination: Path) -> None:
    marker = destination / ".git"
    if not marker.exists():
        if destination.exists():
            raise RuntimeError(f"Dependency path exists but is not a git checkout: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", repo, str(destination)], check=True
        )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )
    found = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if found != revision:
        raise RuntimeError(f"Dependency revision mismatch: {found} != {revision}")


def materialize(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    vjepa = config["vjepa21"]
    vjepa_dependency = project_path(vjepa["dependency_root"])
    if not (vjepa_dependency / "app" / "vjepa_2_1" / "models").is_dir():
        raise RuntimeError(
            f"V-JEPA 2 dependency is missing at {vjepa_dependency}. "
            "Run bash setup_env_uv.sh first; it applies a required dtype compatibility patch."
        )
    download_url(vjepa["checkpoint_url"], project_path(vjepa["checkpoint_path"]))

    decoder = config["jepa_wms_decoder"]
    ensure_dependency(
        decoder["dependency_repo"],
        decoder["dependency_revision"],
        project_path(decoder["dependency_root"]),
    )
    from huggingface_hub import hf_hub_download

    cached = Path(
        hf_hub_download(
            repo_id=decoder["model_id"], filename=decoder["checkpoint_file"]
        )
    )
    destination = project_path(decoder["checkpoint_path"])
    materialize(cached, destination)
    print(f"V-JEPA 2.1 checkpoint: {project_path(vjepa['checkpoint_path'])}")
    print(f"JEPA-WMs decoder: {destination}")
    print(f"JEPA-WMs source: {project_path(decoder['dependency_root'])}")


if __name__ == "__main__":
    main()
