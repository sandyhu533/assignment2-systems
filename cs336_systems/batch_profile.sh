#!/usr/bin/env bash
# Run nsys profile sweep for CS336 Assignment 2.
#
# Two sweep targets (select with SWEEP env var):
#   - model     : end-to-end transformer benchmark   (cs336_systems/benchmark.py)
#   - attention : scaled dot-product attention bench  (cs336_systems/att_benchmark.py)
#                 d_model        in {16, 32, 64, 128}
#                 sequence_length in {256, 1024, 4096, 8192, 16384}
#   - all       : model sweep, then attention sweep
#
# Common axes:
#   - modes:     forward / forward_backward / full
#   - precision: fp32 / amp (BF16 autocast)
#
# Memory snapshots:
#   When MEM_SNAPSHOT=1 (default), each run is passed
#   --mem_snapshot <out>_mem.pickle so benchmark.py can dump a
#   torch.cuda memory snapshot (view at https://pytorch.org/memory_viz).
#
# Outputs land in report/<timestamp>/ as:
#   - <tag>.nsys-rep        nsys trace
#   - <tag>.log             stdout/stderr
#   - <tag>_mem.pickle      memory snapshot (if MEM_SNAPSHOT=1)
#   - timings.csv           aggregated benchmark timings
#
# Usage:
#   bash nsys_sweep.sh                      # attention sweep (default)
#   SWEEP=model bash nsys_sweep.sh          # model size sweep
#   SWEEP=all   bash nsys_sweep.sh          # both
#   DRY_RUN=1   bash nsys_sweep.sh          # print without running
#   MEM_SNAPSHOT=0 bash nsys_sweep.sh       # skip memory snapshots
#   OUT_DIR=report/my-run bash nsys_sweep.sh
 
set -euo pipefail
 
DEFAULT_OUT_DIR="report/$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
WARMUP="${WARMUP:-5}"
STEPS="${STEPS:-10}"
 
# Which sweep to run: attention | model | all
SWEEP="${SWEEP:-attention}"
# Pass --mem_snapshot <path> to benchmark.py when set to 1.
MEM_SNAPSHOT="${MEM_SNAPSHOT:-1}"
 
# Benchmark entry points (one per sweep).
MODEL_BENCH_MODULE="${MODEL_BENCH_MODULE:-cs336_systems/benchmark.py}"
ATTN_BENCH_MODULE="${ATTN_BENCH_MODULE:-cs336_systems/att_benchmark.py}"
 
mkdir -p "$OUT_DIR"
 
CSV_FILE="${OUT_DIR}/timings.csv"
 
# ---- attention sweep axes ----
ATTN_D_MODELS=(16 32 64 128)
ATTN_SEQ_LENS=(256 1024 4096 8192 16384)
ATTN_MODE="${ATTN_MODE:-forward_backward}"
ATTN_PRECISION="${ATTN_PRECISION:-fp32}"
 
# ---- model sweep configs ----
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
 
# CSV header (added bench, d_model, snapshot columns).
echo "timestamp,bench,size,d_model,ctx,mode,precision,warmup,steps,status,mean_ms,std_ms,median_ms,min_ms,p95_ms,peak_mem_gb,peak_resv_gb,tokens_per_sec,tflops,mfu,snapshot,error" > "$CSV_FILE"
 
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
  local bench="$1" size="$2" d_model="$3" ctx="$4" mode="$5" precision="$6" status="$7" json_or_err="$8" snapshot="${9:-}"
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
 
  echo "${ts},${bench},${size},${d_model},${ctx},${mode},${precision},${WARMUP},${STEPS},${status},${mean_ms},${std_ms},${median_ms},${min_ms},${p95_ms},${peak_mem},${peak_resv},${tps},${tflops},${mfu},${snapshot},${error}" >> "$CSV_FILE"
}
 
# Wrap a benchmark invocation in nsys and record its result.
#   $1 = module (python entry point to run)
#   $2 = tag (used for output paths)
#   $3 = bench label for CSV ("model" | "attention")
#   $4 = size  (model sweep) / "" for attention
#   $5 = d_model ("" for model sweep)
#   $6 = ctx / seq_len
#   $7 = mode
#   $8 = precision
#   shift 8; remaining args = benchmark args
run_profiled() {
  local module="$1" tag="$2" bench="$3" size="$4" d_model="$5" ctx="$6" mode="$7" precision="$8"
  shift 8
  local bench_args=( "$@" )
 
  local out="${OUT_DIR}/${tag}"
  local log="${OUT_DIR}/${tag}.log"
  local snap=""
  if [[ "$MEM_SNAPSHOT" == "1" ]]; then
    snap="${out}_mem.pickle"
    bench_args+=( --mem_snapshot "${snap}" )
  fi
 
  echo ""
  echo "========================================================"
  echo "[$(date +%H:%M:%S)] ${bench}: ${tag}  ->  ${out}.nsys-rep"
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
      python "${module}"
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
      append_csv_row "$bench" "$size" "$d_model" "$ctx" "$mode" "$precision" "ok" "$json_line" "$snap"
      local mean_ms peak_mem
      mean_ms=$(json_get "$json_line" mean_ms)
      peak_mem=$(json_get "$json_line" peak_mem_gb)
      printf '  ✓ mean=%s ms, peak_mem=%s GB\n' "$mean_ms" "$peak_mem"
    else
      echo "  ⚠ no RESULT_JSON in output (check $log)" >&2
      append_csv_row "$bench" "$size" "$d_model" "$ctx" "$mode" "$precision" "no_result" "missing RESULT_JSON in stdout" "$snap"
    fi
  else
    local rc=$?
    local err_tail
    err_tail=$(tail -3 "$log" | tr '\n' ' ')
    echo "  ⚠ FAILED rc=$rc (see $log): ${err_tail}" >&2
    append_csv_row "$bench" "$size" "$d_model" "$ctx" "$mode" "$precision" "rc=$rc" "$err_tail" "$snap"
    return 0
  fi
}
 
run_model_one() {
  local size="$1" ctx="$2" mode="$3" precision="$4"
  local tag="${size}_ctx${ctx}_${mode}_${precision}"
 
  # ASSUMPTION: model benchmark CLI = --size / --mode / --context_length (unchanged from original).
  local bench_args=(
    --size "${size}"
    --mode "${mode}"
    --context_length "${ctx}"
    --warmup "${WARMUP}"
    --steps  "${STEPS}"
  )
  [[ "$precision" == "amp" ]] && bench_args+=( --use_amp )
 
  run_profiled "$MODEL_BENCH_MODULE" "$tag" "model" "$size" "" "$ctx" "$mode" "$precision" "${bench_args[@]}"
}
 
run_attention_one() {
  local d="$1" seq="$2" mode="$3" precision="$4"
  local tag="attn_d${d}_seq${seq}_${mode}_${precision}"
 
  # ASSUMPTIONS about att_benchmark.py's CLI — adjust to match your argparse:
  #   --d_model         head/model dim
  #   --context_length  sequence length
  local bench_args=(
    --d_model "${d}"
    --context_length "${seq}"
    --mode "${mode}"
    --warmup "${WARMUP}"
    --steps  "${STEPS}"
  )
  [[ "$precision" == "amp" ]] && bench_args+=( --use_amp )
 
  run_profiled "$ATTN_BENCH_MODULE" "$tag" "attention" "" "$d" "$seq" "$mode" "$precision" "${bench_args[@]}"
}
 
run_model_sweep() {
  echo ""
  echo "=== Model sweep: ${#CONFIGS[@]} runs ==="
  local i=0
  for entry in "${CONFIGS[@]}"; do
    i=$((i + 1))
    echo ""
    echo "### Model progress: $i / ${#CONFIGS[@]} ###"
    # shellcheck disable=SC2086
    set -- $entry
    run_model_one "$1" "$2" "$3" "$4"
  done
}
 
run_attention_sweep() {
  local total=$(( ${#ATTN_D_MODELS[@]} * ${#ATTN_SEQ_LENS[@]} ))
  echo ""
  echo "=== Attention sweep: ${total} runs (mode=${ATTN_MODE}, prec=${ATTN_PRECISION}) ==="
  local i=0
  for d in "${ATTN_D_MODELS[@]}"; do
    for s in "${ATTN_SEQ_LENS[@]}"; do
      i=$((i + 1))
      echo ""
      echo "### Attention progress: $i / ${total} ###"
      run_attention_one "$d" "$s" "$ATTN_MODE" "$ATTN_PRECISION"
    done
  done
}
 
echo "Output dir:    ${OUT_DIR}/"
echo "Sweep:         ${SWEEP}"
echo "mem_snapshot:  ${MEM_SNAPSHOT}"
echo "CSV:           ${CSV_FILE}"
 
case "$SWEEP" in
  attention) run_attention_sweep ;;
  model)     run_model_sweep ;;
  all)       run_model_sweep; run_attention_sweep ;;
  *) echo "Unknown SWEEP='${SWEEP}' (use: attention | model | all)" >&2; exit 1 ;;
esac
 
echo ""
echo "========================================================"
echo "Done. Reports in ${OUT_DIR}/:"
ls -lh "${OUT_DIR}"/*.nsys-rep 2>/dev/null || echo "  (no nsys reports generated)"
if [[ "$MEM_SNAPSHOT" == "1" ]]; then
  echo ""
  echo "Memory snapshots:"
  ls -lh "${OUT_DIR}"/*_mem.pickle 2>/dev/null || echo "  (no snapshots generated)"
fi
echo ""
echo "Timings CSV: ${CSV_FILE}"
if [[ -f "$CSV_FILE" ]]; then
  echo ""
  echo "--- Summary ---"
  column -t -s ',' "$CSV_FILE" | head -30
fi