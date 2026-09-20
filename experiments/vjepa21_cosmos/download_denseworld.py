"""Reuse the source-disjoint DenseWorld subset downloader."""
from experiments.jepa_cosmos import download_denseworld as implementation
from experiments.vjepa21_cosmos.common import load_config


def main() -> None:
    implementation.load_config = load_config
    implementation.main()


if __name__ == "__main__":
    main()
