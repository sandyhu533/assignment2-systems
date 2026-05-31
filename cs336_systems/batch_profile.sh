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
STEPS="${STEPS:-100}"

# Which sweep to run: attention | model | all
SWEEP="${SWEEP:-attention}"
# Pass --mem_snapshot <path> to benchmark.py when set to 1.
MEM_SNAPSHOT="${MEM_SNAPSHOT:-1}"

# Benchmark entry points (one per sweep).
MODEL_BENCH_MODULE="${MODEL_BENCH_MODULE:-cs336_systems/benchmark.py}"
ATTN_BENCH_MODULE="${ATTN_BENCH_MODULE:-cs336_systems/att_benchmark.py}"
x t
mkdir -p "$OUT_DIR"

CSV_FILE="${OUT_DIR}/timings.csv"
# JSONL is the source of truth: one full JSON record appended per run (crash-safe).
# timings.csv is regenerated from it at the end, with columns derived from the data.
JSONL_FILE="${OUT_DIR}/timings.jsonl"
# run.log records this sweep's invocation/config (header) + the final results table.
RUN_LOG="${OUT_DIR}/run.log"
export CSV_FILE JSONL_FILE
: > "$JSONL_FILE"

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

# No upfront CSV header: columns aren't known until we see what the benchmark
# reports. The CSV is generated from the JSONL at the end (see write_csv).

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

# Append one run's record to the JSONL file. Every key the benchmark reports in
# RESULT_JSON becomes part of the record — nothing about the stats is hardcoded.
# Bash-side metadata wins on name collisions; all other payload keys are added.
record_result() {
  local bench="$1" size="$2" d_model="$3" ctx="$4" mode="$5" precision="$6" status="$7" json_or_err="$8" snapshot="${9:-}"
  local ts; ts="$(date +%Y-%m-%d_%H:%M:%S)"

  TS="$ts" BENCH="$bench" SIZE="$size" DMODEL="$d_model" CTX="$ctx" \
  MODE="$mode" PREC="$precision" RWARMUP="$WARMUP" RSTEPS="$STEPS" \
  STATUS="$status" SNAP="$snapshot" PAYLOAD="$json_or_err" \
  python3 -c '
import json, os

row = {
    "timestamp": os.environ["TS"],
    "bench":     os.environ["BENCH"],
    "size":      os.environ["SIZE"],
    "d_model":   os.environ["DMODEL"],
    "ctx":       os.environ["CTX"],
    "mode":      os.environ["MODE"],
    "precision": os.environ["PREC"],
    "warmup":    os.environ["RWARMUP"],
    "steps":     os.environ["RSTEPS"],
    "status":    os.environ["STATUS"],
    "snapshot":  os.environ["SNAP"],
    "error":     "",
}
payload = os.environ.get("PAYLOAD", "")
if row["status"] == "ok":
    try:
        stats = json.loads(payload)
    except Exception as e:
        row["error"] = "bad RESULT_JSON: %s" % e
    else:
        for k, v in stats.items():          # every reported kv -> a column
            if k not in row:                 # bash meta is authoritative
                row[k] = v
else:
    row["error"] = payload

with open(os.environ["JSONL_FILE"], "a") as f:
    f.write(json.dumps(row) + "\n")
'
}

# Regenerate timings.csv from the JSONL. Column order: fixed meta lead, then any
# extra keys in first-seen order, then snapshot/error. Missing keys -> empty cell.
write_csv() {
  python3 -c '
import json, os, csv

lead  = ["timestamp", "bench", "size", "d_model", "ctx",
         "mode", "precision", "warmup", "steps", "status"]
trail = ["snapshot", "error"]
fixed = set(lead) | set(trail)

rows = []
with open(os.environ["JSONL_FILE"]) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

dyn, seen = [], set(fixed)
for r in rows:
    for k in r:
        if k not in seen:
            seen.add(k); dyn.append(k)

header = lead + dyn + trail
with open(os.environ["CSV_FILE"], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in header})
'
}

# Write the run header (invocation + resolved config + environment) to run.log
# and to the screen. Call with the script's args: write_run_header "$@".
write_run_header() {
  local host nsysver gpu commit
  host="$(hostname 2>/dev/null || true)";                                          [[ -n "$host" ]]    || host="?"
  nsysver="$(nsys --version 2>/dev/null | head -1 || true)";                        [[ -n "$nsysver" ]] || nsysver="n/a"
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd'; ' - || true)"; [[ -n "$gpu" ]] || gpu="n/a"
  commit="$(git rev-parse --short HEAD 2>/dev/null || true)";                       [[ -n "$commit" ]]  || commit="n/a"

  {
    echo "============================================================"
    echo "# nsys_sweep run log"
    echo "timestamp:    $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "host:         ${host}"
    echo "cwd:          $(pwd)"
    echo "invocation:   SWEEP=${SWEEP} MEM_SNAPSHOT=${MEM_SNAPSHOT} WARMUP=${WARMUP} STEPS=${STEPS} OUT_DIR=${OUT_DIR} bash $0 $*"
    echo "------------------------------------------------------------"
    echo "out_dir:      ${OUT_DIR}"
    echo "sweep:        ${SWEEP}"
    echo "mem_snapshot: ${MEM_SNAPSHOT}"
    echo "warmup/steps: ${WARMUP} / ${STEPS}"
    echo "model module: ${MODEL_BENCH_MODULE}"
    echo "attn module:  ${ATTN_BENCH_MODULE}"
    echo "attn axes:    d_model={${ATTN_D_MODELS[*]}} seq={${ATTN_SEQ_LENS[*]}} mode=${ATTN_MODE} prec=${ATTN_PRECISION}"
    echo "nsys:         ${nsysver}"
    echo "gpu:          ${gpu}"
    echo "git commit:   ${commit}"
    echo "============================================================"
  } | tee "$RUN_LOG"
}

# Append the results table and output pointers to run.log and the screen.
write_run_results() {
  {
    echo ""
    echo "============================================================"
    echo "Done. Output dir: ${OUT_DIR}/"
    echo ""
    echo "nsys reports:"
    ls -1 "${OUT_DIR}"/*.nsys-rep 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    if [[ "$MEM_SNAPSHOT" == "1" ]]; then
      echo "memory snapshots:"
      ls -1 "${OUT_DIR}"/*_mem.pickle 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    fi
    echo ""
    echo "JSONL: ${JSONL_FILE}"
    echo "CSV:   ${CSV_FILE}"
    echo ""
    echo "--- Results ---"
    if [[ -f "$CSV_FILE" ]]; then
      column -t -s ',' "$CSV_FILE" 2>/dev/null || cat "$CSV_FILE"
    else
      echo "(no CSV generated)"
    fi
    echo "============================================================"
  } | tee -a "$RUN_LOG"
}
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
      record_result "$bench" "$size" "$d_model" "$ctx" "$mode" "$precision" "ok" "$json_line" "$snap"
      local mean_ms peak_mem
      mean_ms=$(json_get "$json_line" mean_ms)
      peak_mem=$(json_get "$json_line" peak_mem_gb)
      printf '  ✓ mean=%s ms, peak_mem=%s GB\n' "$mean_ms" "$peak_mem"
    else
      echo "  ⚠ no RESULT_JSON in output (check $log)" >&2
      record_result "$bench" "$size" "$d_model" "$ctx" "$mode" "$precision" "no_result" "missing RESULT_JSON in stdout" "$snap"
    fi
  else
    local rc=$?
    local err_tail
    err_tail=$(tail -3 "$log" | tr '\n' ' ')
    echo "  ⚠ FAILED rc=$rc (see $log): ${err_tail}" >&2
    record_result "$bench" "$size" "$d_model" "$ctx" "$mode" "$precision" "rc=$rc" "$err_tail" "$snap"
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

write_run_header "$@"

case "$SWEEP" in
  attention) run_attention_sweep ;;
  model)     run_model_sweep ;;
  all)       run_model_sweep; run_attention_sweep ;;
  *) echo "Unknown SWEEP='${SWEEP}' (use: attention | model | all)" >&2; exit 1 ;;
esac

# Build timings.csv from the JSONL (columns derived from whatever was reported).
if [[ "${DRY_RUN:-0}" != "1" ]] && [[ -s "$JSONL_FILE" ]]; then
  write_csv
fi

write_run_results