#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/run_vjepa21_jepawms.sh <validate|data|assets|cache|train|infer> <config> [extra args...]"
    exit 2
fi

STAGE="$1"
CONFIG="$2"
shift 2

case "$STAGE" in
    validate)
        python -m experiments.vjepa21_jepawms.validate_setup --config "$CONFIG" "$@"
        ;;
    data)
        python -m experiments.vjepa21_jepawms.download_denseworld --config "$CONFIG" "$@"
        ;;
    assets)
        python -m experiments.vjepa21_jepawms.download_assets --config "$CONFIG" "$@"
        ;;
    cache)
        python -m experiments.vjepa21_jepawms.prepare_cache --config "$CONFIG" "$@"
        ;;
    train)
        python -m experiments.vjepa21_jepawms.train_adapter --config "$CONFIG" "$@"
        ;;
    infer)
        python -m experiments.vjepa21_jepawms.infer --config "$CONFIG" "$@"
        ;;
    *)
        echo "Unknown stage: $STAGE"
        exit 2
        ;;
esac
