#!/usr/bin/env bash
# Compare distributed training modes on a single node, 2 GPUs, with nsys-based
# gradient-communication measurement.
#
# For each --dist_mode, runs the benchmark under nsys, then extracts total NCCL
# kernel time (AllReduce / ReduceScatter / AllGather) from the trace and reports
# it alongside per-step time. This is the reliable way to measure comm time for
# overlap_ddp / zero1 / fsdp, where wall-clock subtraction breaks down.
#
# Multi-process note: mp.spawn children are traced via nsys --output=...%p, so
# each rank writes its own .nsys-rep. We aggregate NCCL time from the child
# trace with the most NCCL activity (the parent process has none).
#
# Usage:
#   bash dist_sweep_nsys.sh                          # all modes, small, amp+compile
#   SIZE=medium bash dist_sweep_nsys.sh
#   MODES="naive_ddp zero1 fsdp" bash dist_sweep_nsys.sh
#   USE_AMP=0 COMPILE=0 bash dist_sweep_nsys.sh      # fp32, no compile
#   DRY_RUN=1 bash dist_sweep_nsys.sh                # print commands only
#   NO_NSYS=1 bash dist_sweep_nsys.sh                # timing only, skip nsys

set -euo pipefail

DEFAULT_OUT_DIR="report/dist_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
WARMUP="${WARMUP:-5}"
STEPS="${STEPS:-20}"

BENCH_MODULE="${BENCH_MODULE:-cs336_systems/benchmark.py}"

MODES="${MODES:-naive_ddp flatten_ddp overlap_ddp zero1 fsdp}"

SIZE="${SIZE:-small}"
CTX="${CTX:-512}"
MODE="${MODE:-full}"          # comm only matters with a backward pass

USE_AMP="${USE_AMP:-1}"
COMPILE="${COMPILE:-1}"

MEM_SNAPSHOT="${MEM_SNAPSHOT:-0}"   # off by default; nsys + mem snapshot both heavy
NO_NSYS="${NO_NSYS:-0}"             # set 1 to skip profiling entirely

mkdir -p "$OUT_DIR"
CSV_FILE="${OUT_DIR}/timings.csv"
JSONL_FILE="${OUT_DIR}/timings.jsonl"
RUN_LOG="${OUT_DIR}/run.log"
export CSV_FILE JSONL_FILE
: > "$JSONL_FILE"

# ---------------------------------------------------------------------------
# Resolve the nsys stats report name once (varies across nsys versions).
# Newer: cuda_gpu_kern_sum   Older: gpukernsum
# ---------------------------------------------------------------------------
KERN_REPORT=""
resolve_kern_report() {
  [[ "$NO_NSYS" == "1" ]] && return 0
  if ! command -v nsys >/dev/null 2>&1; then
    echo "  ⚠ nsys not found; falling back to timing-only (NO_NSYS=1)" >&2
    NO_NSYS=1
    return 0
  fi
  local avail
  avail="$(nsys stats --help-reports 2>/dev/null || true)"
  if grep -q 'cuda_gpu_kern_sum' <<<"$avail"; then
    KERN_REPORT="cuda_gpu_kern_sum"
  elif grep -q 'gpukernsum' <<<"$avail"; then
    KERN_REPORT="gpukernsum"
  else
    # Last resort: try the new name, let it fail loudly per-run.
    KERN_REPORT="cuda_gpu_kern_sum"
    echo "  ⚠ could not confirm nsys kern-sum report name; assuming ${KERN_REPORT}" >&2
  fi
}

# ---------------------------------------------------------------------------
# Extract total NCCL kernel time (ns) from a single .nsys-rep.
# Sums "Total Time" over any kernel whose name matches NCCL collectives.
# Robust to column reordering by reading the CSV header.
# ---------------------------------------------------------------------------
nccl_ns_from_rep() {
  local rep="$1"
  nsys stats --report "$KERN_REPORT" --format csv --force-export=true "$rep" 2>/dev/null \
  | python3 -c '
import csv, sys, re
rdr = csv.reader(sys.stdin)
header = None
total = 0.0
name_idx = time_idx = None
for row in rdr:
    if not row:
        continue
    if header is None:
        header = [c.strip().lower() for c in row]
        # find a name-ish column and a total-time column
        for i, c in enumerate(header):
            if name_idx is None and ("name" in c or "kernel" in c):
                name_idx = i
            if time_idx is None and "total time" in c:
                time_idx = i
        if name_idx is None or time_idx is None:
            # header not as expected; bail with 0
            print(0); sys.exit(0)
        continue
    if name_idx >= len(row) or time_idx >= len(row):
        continue
    name = row[name_idx]
    if re.search(r"nccl|allreduce|reducescatter|allgather|broadcast", name, re.I):
        raw = row[time_idx].replace(",", "").strip()
        try:
            total += float(raw)
        except ValueError:
            pass
print(int(total))
'
}

# Pick the child trace with the most NCCL activity (parent has ~none) and
# return its NCCL total in ns. Scans all %p-suffixed reps for this tag.
nccl_ns_for_tag() {
  local tag="$1"
  local best=0
  shopt -s nullglob
  for rep in "${OUT_DIR}/${tag}"_*.nsys-rep; do
    local ns
    ns="$(nccl_ns_from_rep "$rep")"
    [[ -z "$ns" ]] && ns=0
    if (( ns > best )); then best=$ns; fi
  done
  shopt -u nullglob
  echo "$best"
}

# ---------------------------------------------------------------------------
# JSONL record (dynamic columns; bash meta authoritative)
# ---------------------------------------------------------------------------
record_result() {
  local dist_mode="$1" size="$2" ctx="$3" step_mode="$4" precision="$5" \
        compile="$6" status="$7" json_or_err="$8" nccl_ms="${9:-}"
  local ts; ts="$(date +%Y-%m-%d_%H:%M:%S)"

  TS="$ts" DMODE="$dist_mode" SIZE="$size" CTX="$ctx" STEPMODE="$step_mode" \
  PREC="$precision" COMPILE="$compile" RWARMUP="$WARMUP" RSTEPS="$STEPS" \
  STATUS="$status" NCCLMS="$nccl_ms" PAYLOAD="$json_or_err" \
  python3 -c '
import json, os
row = {
    "timestamp":     os.environ["TS"],
    "dist_mode":     os.environ["DMODE"],
    "size":          os.environ["SIZE"],
    "ctx":           os.environ["CTX"],
    "step_mode":     os.environ["STEPMODE"],
    "precision":     os.environ["PREC"],
    "compile":       os.environ["COMPILE"],
    "warmup":        os.environ["RWARMUP"],
    "steps":         os.environ["RSTEPS"],
    "status":        os.environ["STATUS"],
    "nccl_total_ms": os.environ.get("NCCLMS", ""),
    "error":         "",
}
payload = os.environ.get("PAYLOAD", "")
if row["status"] == "ok":
    try:
        stats = json.loads(payload)
    except Exception as e:
        row["error"] = "bad RESULT_JSON: %s" % e
    else:
        for k, v in stats.items():
            if k not in row:
                row[k] = v
else:
    row["error"] = payload
with open(os.environ["JSONL_FILE"], "a") as f:
    f.write(json.dumps(row) + "\n")
'
}

# ---------------------------------------------------------------------------
# CSV from JSONL, with two derived columns:
#   nccl_per_step_ms = nccl_total_ms / steps
#   nccl_comm_pct    = nccl_per_step_ms / mean_ms * 100
# ---------------------------------------------------------------------------
write_csv() {
  python3 -c '
import json, os, csv

lead  = ["timestamp", "dist_mode", "size", "ctx", "step_mode",
         "precision", "compile", "warmup", "steps", "status"]
trail = ["error"]
fixed = set(lead) | set(trail)

rows = []
with open(os.environ["JSONL_FILE"]) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        def fnum(k):
            try:
                v = float(str(r.get(k, "")).replace(",", ""))
                return v if v == v else None
            except Exception:
                return None
        nccl_total = fnum("nccl_total_ms")
        steps      = fnum("steps")
        mean       = fnum("mean_ms")
        if nccl_total is not None and steps and steps > 0:
            per = nccl_total / steps
            r["nccl_per_step_ms"] = round(per, 3)
            if mean and mean > 0:
                r["nccl_comm_pct"] = round(per / mean * 100, 2)
        rows.append(r)

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

# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
run_one() {
  local dist_mode="$1"
  local prec="fp32"; [[ "$USE_AMP" == "1" ]] && prec="amp"
  local comp="no";   [[ "$COMPILE"  == "1" ]] && comp="yes"
  local tag="${dist_mode}_${SIZE}_ctx${CTX}_${MODE}_${prec}"
  [[ "$comp" == "yes" ]] && tag="${tag}_compile"

  local log="${OUT_DIR}/${tag}.log"

  local bench_args=(
    --size "$SIZE"
    --mode "$MODE"
    --context_length "$CTX"
    --warmup "$WARMUP"
    --steps  "$STEPS"
    --dist
    --dist_mode "$dist_mode"
  )
  [[ "$USE_AMP" == "1" ]] && bench_args+=( --use_amp )
  [[ "$COMPILE" == "1" ]] && bench_args+=( --compile )
  if [[ "$MEM_SNAPSHOT" == "1" ]]; then
    bench_args+=( --mem_snapshot "${OUT_DIR}/${tag}_mem.pickle" )
  fi

  local cmd
  if [[ "$NO_NSYS" == "1" ]]; then
    cmd=( uv run python "$BENCH_MODULE" "${bench_args[@]}" )
  else
    # %p -> one trace per process; child ranks carry the NCCL activity.
    cmd=( uv run nsys profile
            --trace=cuda,cudnn,cublas,nvtx
            --capture-range=cudaProfilerApi
            --capture-range-end=stop
            --output="${OUT_DIR}/${tag}_%p"
            --force-overwrite=true
            -- python "$BENCH_MODULE" "${bench_args[@]}" )
  fi

  echo ""
  echo "========================================================"
  echo "[$(date +%H:%M:%S)] dist_mode=${dist_mode}  ->  ${tag}"
  echo "========================================================"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '  %s\n' "${cmd[*]}"
    return 0
  fi

  if "${cmd[@]}" > "$log" 2>&1; then
    local json_line
    json_line=$(grep -E '^RESULT_JSON ' "$log" | tail -1 | sed 's/^RESULT_JSON //')
    if [[ -z "$json_line" ]]; then
      echo "  ⚠ no RESULT_JSON (check $log)" >&2
      record_result "$dist_mode" "$SIZE" "$CTX" "$MODE" "$prec" "$comp" "no_result" "missing RESULT_JSON" ""
      return 0
    fi

    # Extract NCCL kernel time from the trace(s).
    local nccl_ms=""
    if [[ "$NO_NSYS" != "1" ]]; then
      local nccl_ns
      nccl_ns="$(nccl_ns_for_tag "$tag")"
      if [[ -n "$nccl_ns" && "$nccl_ns" != "0" ]]; then
        nccl_ms="$(python3 -c "print(round($nccl_ns/1e6, 3))")"
      else
        echo "  ⚠ no NCCL kernels found in trace for ${tag}" >&2
      fi
    fi

    record_result "$dist_mode" "$SIZE" "$CTX" "$MODE" "$prec" "$comp" "ok" "$json_line" "$nccl_ms"
    local mean
    mean=$(python3 -c "import json;print(json.loads('''$json_line''').get('mean_ms',''))" 2>/dev/null)
    printf '  ✓ mean=%s ms, nccl_total=%s ms\n' "$mean" "${nccl_ms:-n/a}"
  else
    local rc=$?
    local err_tail; err_tail=$(tail -3 "$log" | tr '\n' ' ')
    echo "  ⚠ FAILED rc=$rc (see $log): ${err_tail}" >&2
    record_result "$dist_mode" "$SIZE" "$CTX" "$MODE" "$prec" "$comp" "rc=$rc" "$err_tail" ""
    return 0
  fi
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
resolve_kern_report
{
  echo "============================================================"
  echo "# dist_sweep_nsys run log"
  echo "timestamp:    $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "host:         $(hostname 2>/dev/null || echo '?')"
  echo "cwd:          $(pwd)"
  echo "out_dir:      ${OUT_DIR}"
  echo "modes:        ${MODES}"
  echo "size/ctx:     ${SIZE} / ${CTX}   step_mode=${MODE}"
  echo "amp/compile:  ${USE_AMP} / ${COMPILE}"
  echo "warmup/steps: ${WARMUP} / ${STEPS}"
  echo "nsys:         $([[ "$NO_NSYS" == "1" ]] && echo disabled || echo "enabled (report=${KERN_REPORT})")"
  echo "nsys version: $(nsys --version 2>/dev/null | head -1 || echo n/a)"
  echo "gpu:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd'; ' - || echo n/a)"
  echo "============================================================"
} | tee "$RUN_LOG"

i=0; total=$(wc -w <<< "$MODES")
for dm in $MODES; do
  i=$((i+1))
  echo ""
  echo "### progress: $i / $total ###"
  run_one "$dm"
done

if [[ "${DRY_RUN:-0}" != "1" ]] && [[ -s "$JSONL_FILE" ]]; then
  write_csv
fi

{
  echo ""
  echo "============================================================"
  echo "Done. Output dir: ${OUT_DIR}/"
  echo "CSV: ${CSV_FILE}"
  echo ""
  echo "--- Results ---"
  if [[ -f "$CSV_FILE" ]]; then
    column -t -s ',' "$CSV_FILE" 2>/dev/null || cat "$CSV_FILE"
  fi
  echo "============================================================"
} | tee -a "$RUN_LOG"