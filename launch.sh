#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gpu="$(nvidia-smi -i "${ATTENTION_GPU:?Set ATTENTION_GPU to an idle GPU index or UUID}" --query-gpu=uuid --format=csv,noheader)"
[[ "$gpu" == GPU-* && "$gpu" != *$'\n'* ]] || { echo 'Select exactly one GPU' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$PWD:$PWD/upstream:$PWD/upstream/zoology${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
mkdir -p logs runs
# A cooperative device lock plus process checks; no other workload is stopped.
exec flock -n "${ATTENTION_LOCK_DIR:-/tmp}/attention-${gpu}.lock" \
  "${ATTENTION_PYTHON:-$PWD/.venv/bin/python}" campaign.py "$@"
