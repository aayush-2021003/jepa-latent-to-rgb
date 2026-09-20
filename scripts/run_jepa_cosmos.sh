#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/run_jepa_cosmos.sh <validate|data|assets|cache|train|infer> <config> [extra args...]"
    exit 2
fi

STAGE="$1"
CONFIG="$2"
shift 2

case "$STAGE" in
    validate)
        python -m experiments.jepa_cosmos.validate_setup --config "$CONFIG" "$@"
        ;;
    data)
        python -m experiments.jepa_cosmos.download_denseworld --config "$CONFIG" "$@"
        ;;
    assets)
        python -m experiments.jepa_cosmos.download_assets --config "$CONFIG" "$@"
        ;;
    cache)
        python -m experiments.jepa_cosmos.prepare_latents --config "$CONFIG" "$@"
        ;;
    train)
        python -m experiments.jepa_cosmos.train_adapter --config "$CONFIG" "$@"
        ;;
    infer)
        python -m experiments.jepa_cosmos.infer --config "$CONFIG" "$@"
        ;;
    *)
        echo "Unknown stage: $STAGE"
        exit 2
        ;;
esac

