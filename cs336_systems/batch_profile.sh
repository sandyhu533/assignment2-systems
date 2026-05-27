#!/usr/bin/env bash
# Run nsys profile sweep for CS336 Assignment 2.
#
# Configurations:
#   1. size sweep:           small  + medium  @ ctx=512
#   2. context_length sweep: small  @ ctx=256 / 512 / 1024
#   3. modes:                forward / forward_backward / full
#   4. precision:            fp32 / amp (BF16 autocast)
#
# Outputs land in nsys-report/<timestamp>/ as:
#   - <size>_ctx<ctx>_<mode>_<precision>.nsys-rep
#   - timings.csv  (aggregated benchmark timings, includes precision column)
#
# Usage:
#   bash nsys_sweep.sh                 # run everything
#   DRY_RUN=1 bash nsys_sweep.sh       # print without running
#   OUT_DIR=nsys-report/my-run bash nsys_sweep.sh

set -euo pipefail

DEFAULT_OUT_DIR="nsys-report/$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
WARMUP="${WARMUP:-5}"
STEPS="${STEPS:-10}"

mkdir -p "$OUT_DIR"

CSV_FILE="${OUT_DIR}/timings.csv"

# Configs as "size ctx mode precision" 4-tuples.
# precision: "fp32" (no flag) or "amp" (adds --use_amp).
CONFIGS=(
  # --- forward only ---
  # "small  512  forward            fp32"
  # "small  512  forward            amp"
  # "medium 512  forward            fp32"
  # "medium 512  forward            amp"
  # "large 512  forward             fp32"
  # "large 512  forward             amp"
  "xl 512  forward             amp"
  # "small  256  forward            fp32"
  # "small  256  forward            amp"
  # "small  1024 forward            fp32"
  # "small  1024 forward            amp"

  # --- full step (fwd + bwd + opt) ---
  # "small  512  full               fp32"
  # "small  512  full               amp"
  # "medium 512  full               fp32"
  # "medium 512  full               amp"
  # "large  512  full               fp32"
  # "large  512  full               amp"
  # "small  256  full               fp32"
  # "small  256  full               amp"
  # "small  1024 full               fp32"
  # "small  1024 full               amp"

  # --- forward + backward (no optimizer step) ---
  # "small  512  forward_backward   fp32"
  # "small  512  forward_backward   amp"
  # "medium 512  forward_backward   fp32"
  # "medium 512  forward_backward   amp"
  # "large  512  forward_backward   fp32"
  # "large  512  forward_backward   amp"
  # "small  256  forward_backward   fp32"
  # "small  256  forward_backward   amp"
  # "small  1024 forward_backward   fp32"
  # "small  1024 forward_backward   amp"
)

# CSV header (added precision column after mode).
echo "timestamp,size,ctx,mode,precision,warmup,steps,status,mean_ms,std_ms,median_ms,min_ms,p95_ms,peak_mem_gb,peak_resv_gb,tokens_per_sec,tflops,mfu,error" > "$CSV_FILE"

json_get() {
  local json="$1"
  local field="$2"
  local default="${3:-}"
  python3 -c "
import json, sys
try:
    d = json.loads('''$json''')
    v = d.get('$field', '$default')
    if v is None or (isinstance(v, float) and v != v):
        print('$default')
    else:
        print(v)
except Exception:
    print('$default')
" 2>/dev/null
}

append_csv_row() {
  local size="$1" ctx="$2" mode="$3" precision="$4" status="$5" json_or_err="$6"
  local ts mean_ms std_ms median_ms min_ms p95_ms peak_mem peak_resv tps tflops mfu error
  ts="$(date +%Y-%m-%d_%H:%M:%S)"

  if [[ "$status" == "ok" ]]; then
    mean_ms=$(json_get   "$json_or_err" mean_ms)
    std_ms=$(json_get    "$json_or_err" std_ms)
    median_ms=$(json_get "$json_or_err" median_ms)
    min_ms=$(json_get    "$json_or_err" min_ms)
    p95_ms=$(json_get    "$json_or_err" p95_ms)
    peak_mem=$(json_get  "$json_or_err" peak_mem_gb)
    peak_resv=$(json_get "$json_or_err" peak_resv_gb)
    tps=$(json_get       "$json_or_err" tokens_per_sec)
    tflops=$(json_get    "$json_or_err" tflops)
    mfu=$(json_get       "$json_or_err" mfu)
    error=""
  else
    mean_ms="" std_ms="" median_ms="" min_ms="" p95_ms=""
    peak_mem="" peak_resv="" tps="" tflops="" mfu=""
    error="$(echo "$json_or_err" | tr ',\n' ';|' | tr -d '"')"
  fi

  echo "${ts},${size},${ctx},${mode},${precision},${WARMUP},${STEPS},${status},${mean_ms},${std_ms},${median_ms},${min_ms},${p95_ms},${peak_mem},${peak_resv},${tps},${tflops},${mfu},${error}" >> "$CSV_FILE"
}

run_one() {
  local size="$1"
  local ctx="$2"
  local mode="$3"
  local precision="$4"     # "fp32" or "amp"
  local tag="${size}_ctx${ctx}_${mode}_${precision}"
  local out="${OUT_DIR}/${tag}"
  local log="${OUT_DIR}/${tag}.log"

  # Build benchmark.py args; add --use_amp only when precision=amp.
  local bench_args=(
    --size "${size}"
    --mode "${mode}"
    --context_length "${ctx}"
    --warmup "${WARMUP}"
    --steps  "${STEPS}"
  )
  if [[ "$precision" == "amp" ]]; then
    bench_args+=( --use_amp )
  fi

  echo ""
  echo "========================================================"
  echo "[$(date +%H:%M:%S)] size=${size}  ctx=${ctx}  mode=${mode}  prec=${precision} →  ${out}.nsys-rep"
  echo "========================================================"

  local cmd=(
    uv run nsys profile
      --trace=cuda,cudnn,cublas,osrt,nvtx
      --cudabacktrace=memory
      --cuda-memory-usage=true
      --pytorch=functions-trace,autograd-shapes-nvtx
      --capture-range=cudaProfilerApi
      --capture-range-end=stop
      --output="${out}"
      --force-overwrite=true
      --
      python cs336_systems/benchmark.py
      "${bench_args[@]}"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '  %s\n' "${cmd[*]}"
    return 0
  fi

  if "${cmd[@]}" > "$log" 2>&1; then
    local json_line
    json_line=$(grep -E '^RESULT_JSON ' "$log" | tail -1 | sed 's/^RESULT_JSON //')
    if [[ -n "$json_line" ]]; then
      append_csv_row "$size" "$ctx" "$mode" "$precision" "ok" "$json_line"
      local mean_ms peak_mem
      mean_ms=$(json_get "$json_line" mean_ms)
      peak_mem=$(json_get "$json_line" peak_mem_gb)
      printf '  ✓ mean=%s ms, peak_mem=%s GB\n' "$mean_ms" "$peak_mem"
    else
      echo "  ⚠ no RESULT_JSON in output (check $log)" >&2
      append_csv_row "$size" "$ctx" "$mode" "$precision" "no_result" "missing RESULT_JSON in stdout"
    fi
  else
    local rc=$?
    local err_tail
    err_tail=$(tail -3 "$log" | tr '\n' ' ')
    echo "  ⚠ FAILED rc=$rc (see $log): ${err_tail}" >&2
    append_csv_row "$size" "$ctx" "$mode" "$precision" "rc=$rc" "$err_tail"
    return 0
  fi
}

echo "Sweep: ${#CONFIGS[@]} runs → ${OUT_DIR}/"
echo "CSV:   ${CSV_FILE}"

i=0
for entry in "${CONFIGS[@]}"; do
  i=$((i + 1))
  echo ""
  echo "### Progress: $i / ${#CONFIGS[@]} ###"
  # shellcheck disable=SC2086
  set -- $entry
  run_one "$1" "$2" "$3" "$4"
done

echo ""
echo "========================================================"
echo "Done. Reports in ${OUT_DIR}/:"
ls -lh "${OUT_DIR}"/*.nsys-rep 2>/dev/null || echo "  (no nsys reports generated)"
echo ""
echo "Timings CSV: ${CSV_FILE}"
if [[ -f "$CSV_FILE" ]]; then
  echo ""
  echo "--- Summary ---"
  column -t -s ',' "$CSV_FILE" | head -30
fi
echo "========================================================"