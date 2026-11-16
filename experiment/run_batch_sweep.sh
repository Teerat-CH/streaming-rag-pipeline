#!/usr/bin/env bash
# Usage (from repo root): bash experiment/run_batch_sweep.sh DOCS LIMIT [OUT]
# Each setting is its own process so no Spark or torch state carries over.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 DOCS LIMIT [OUT]" >&2
    exit 1
fi
DOCS=$1
LIMIT=$2
OUT_ARGS=()
if [[ $# -ge 3 ]]; then
    OUT_ARGS=(--out "$3")
fi

for k in 1 2 4 8; do
    for mode in single spark; do
        python -m experiment.benchmark_batch --docs "$DOCS" --mode "$mode" \
            --k "$k" --limit "$LIMIT" ${OUT_ARGS[@]+"${OUT_ARGS[@]}"}
    done
done
