#!/usr/bin/env bash
# Run nsys profile sweep for CS336 Assignment 2.
#
# Configurations:
#   1. size sweep:           small  + medium  @ ctx=512
#   2. context_length sweep: small  @ ctx=256 / 512 / 1024
#
# Outputs land in nsys-report/<timestamp>/ as <size>_ctx<ctx>.nsys-rep
#
# Usage:
#   bash nsys_sweep.sh                 # run everything
#   DRY_RUN=1 bash nsys_sweep.sh       # print commands without running
#   OUT_DIR=nsys-report/my-run bash nsys_sweep.sh   # custom output dir

set -euo pipefail

# Default: nsys-report/YYYYMMDD_HHMMSS/ for each sweep run.
DEFAULT_OUT_DIR="nsys-report/$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
WARMUP="${WARMUP:-3}"
STEPS="${STEPS:-3}"
MODE="${MODE:-full}"

mkdir -p "$OUT_DIR"

# Configs as "size ctx" pairs. De-dup small/ctx=512 (appears in both sweeps).
CONFIGS=(
  "small  512 forward"
  "medium 512 forward"
  "small  256 forward"
  "small  1024 forward"
  "small  512 full"
  "medium 512 full"
  "small  256 full"
  "small  1024 full"
)

run_one() {
  local size="$1"
  local ctx="$2"
  local mode="$3"
  local tag="${size}_ctx${ctx}_${mode}"
  local out="${OUT_DIR}/${tag}"

  echo ""
  echo "========================================================"
  echo "[$(date +%H:%M:%S)] size=${size}  ctx=${ctx}  mode=${mode} →  ${out}.nsys-rep"
  echo "========================================================"

  local cmd=(
    uv run nsys profile
      --trace=cuda,cudnn,cublas,osrt,nvtx
      --pytorch=functions-trace,autograd-shapes-nvtx
      --capture-range=cudaProfilerApi
      --capture-range-end=stop
      --output="${out}"
      --force-overwrite=true
      --
      python cs336_systems/benchmark.py
        --size "${size}"
        --mode "${mode}"
        --context_length "${ctx}"
        --warmup "${WARMUP}"
        --steps  "${STEPS}"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '  %s\n' "${cmd[*]}"
    return 0
  fi

  if ! "${cmd[@]}"; then
    echo "  ⚠ FAILED: ${tag} (continuing)" >&2
    return 0   # don't abort the whole sweep on a single OOM
  fi
}

echo "Sweep: ${#CONFIGS[@]} runs → ${OUT_DIR}/"
for entry in "${CONFIGS[@]}"; do
  # shellcheck disable=SC2086
  set -- $entry
  run_one "$1" "$2" "$3"
done

echo ""
echo "========================================================"
echo "Done. Reports in ${OUT_DIR}/:"
ls -lh "${OUT_DIR}"/*.nsys-rep 2>/dev/null || echo "  (no reports generated)"
echo "========================================================"