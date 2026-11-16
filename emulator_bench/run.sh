#!/usr/bin/env bash
# StructureFree-DTA Kd paper runs: original fp16 autocast (no --amp). No cache step. train also evaluates.
# Usage: emulator_bench/run.sh [--dry-run] [train|aggregate|all]
# Env: PY GPUS SEEDS RUNS_PER_GPU DATA_ROOT OUT_ROOT EMULATOR_ROOT DRY_RUN=1
# shellcheck disable=SC2086  # PY, GPUS, SEEDS and the task lists split into words on purpose
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EMULATOR_ROOT=${EMULATOR_ROOT:-$(realpath ../..)}
DATA_ROOT=${DATA_ROOT:-$EMULATOR_ROOT/data/processed/datasets}
OUT_ROOT=${OUT_ROOT:-$EMULATOR_ROOT/outputs/baselines/StructureFree-DTA}
PY=${PY:-python}
GPUS=${GPUS:-0}
RUNS_PER_GPU=${RUNS_PER_GPU:-1}
SEEDS=${SEEDS:-"666 777 888"}
STEP=all
for a in "$@"; do
  case $a in
    --dry-run) DRY_RUN=1 ;;
    train|aggregate|all) STEP=$a ;;
    *) echo "usage: $0 [--dry-run] [train|aggregate|all]" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

want train && run $PY -m emulator_bench.run_split_benchmarks --seeds $SEEDS \
  --base_dir "$DATA_ROOT/binding_affinity_dataset/kd" --output_root "$OUT_ROOT/kd" \
  --gpus $GPUS --runs_per_gpu "$RUNS_PER_GPU"
want aggregate && run $PY -m emulator_bench.aggregate_results --output_root "$OUT_ROOT/kd"
exit 0
