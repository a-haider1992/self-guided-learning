#!/bin/bash

set -e

LOGDIR="logs"
mkdir -p "$LOGDIR"

SEEDS=(42 1337 2023 3407 777)

echo "Starting multi-seed experiment..."
echo "=================================="

for S in "${SEEDS[@]}"; do
    echo "Running seed $S"
    if ! python codebase.py --seed "$S" --mode "standard" >> "$LOGDIR/seed_$S.log" 2>&1; then
        echo "Seed $S failed. Stopping execution."
        exit 1
    fi
done

echo "All seeds completed."

