"""Reuse the proven source-disjoint DenseWorld subset downloader."""
from __future__ import annotations

from experiments.jepa_cosmos import download_denseworld as implementation
from experiments.vjepa21_jepawms.common import load_config


def main() -> None:
    # The downloader itself only consumes experiment/data/tracking fields. Replace
    # its schema loader so this experiment does not need fake Cosmos sections.
    implementation.load_config = load_config
    implementation.main()


if __name__ == "__main__":
    main()
