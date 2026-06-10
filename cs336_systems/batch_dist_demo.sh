#!/usr/bin/env bash
#
# All-reduce benchmark sweep.  Self-contained: runs the benchmark across
# {1,10,100,1000 MB} x {2,4,6 ranks}, then aggregates to CSV + plots.
#
# Runner selection:
#   - GPU present  -> "uv run python"
#   - CPU only     -> ".venv/bin/python"
#
# Usage:  ./run_sweep.sh [PY_SCRIPT] [REPORT_ROOT]
# Default: PY_SCRIPT=cs336_systems/dist_single_node.py  REPORT_ROOT=report
# Output goes to REPORT_ROOT/<YYYYMMDD_HHMMSS>/
#
set -uo pipefail

PY_SCRIPT="${1:-cs336_systems/dist_single_node.py}"
REPORT_ROOT="${2:-report}"
VENV_PY="${VENV_PY:-.venv/bin/python}"   # CPU-path interpreter (override via env)

# Each run gets its own subdir keyed by start time: report/YYYYMMDD_HHMMSS/
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${REPORT_ROOT}/${RUN_TS}"
DATA_SIZES_MB=(1 10 100 1000)
WORLD_SIZES=(2 4 6)
WARMUP=5
STEPS=20

RAW_DIR="${OUTDIR}/raw"
mkdir -p "${RAW_DIR}"

# --- pick runner: GPU -> uv run python ; CPU -> .venv/bin/python ---------
# Probe GPU count using uv's env first; if that fails, assume 0 (CPU).
NGPU=$(uv run python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [[ "${NGPU}" -gt 0 ]]; then
  RUNNER=(uv run python)
  MODE="GPU (uv run python)"
else
  if [[ ! -x "${VENV_PY}" ]]; then
    echo "ERROR: no GPU detected and CPU interpreter '${VENV_PY}' not found/executable." >&2
    echo "       create the venv or set VENV_PY=/path/to/python" >&2
    exit 1
  fi
  RUNNER=("${VENV_PY}")
  MODE="CPU (${VENV_PY})"
fi

echo "=== all-reduce sweep ===  (detected ${NGPU} CUDA device(s))"
echo "runner:     ${MODE}"
echo "sizes=${DATA_SIZES_MB[*]}MB  ranks=${WORLD_SIZES[*]}  warmup=${WARMUP} steps=${STEPS}"
echo "output dir: ${OUTDIR}"
echo

for ws in "${WORLD_SIZES[@]}"; do
  if [[ "${NGPU}" -gt 0 && "${ws}" -gt "${NGPU}" ]]; then
    echo ">> SKIP ws=${ws} (only ${NGPU} GPUs)"; continue
  fi
  for mb in "${DATA_SIZES_MB[@]}"; do
    echo -n ">> RUN ws=${ws} mb=${mb} ... "
    "${RUNNER[@]}" "${PY_SCRIPT}" --world_size "${ws}" --data_size_mb "${mb}" \
      --warmup "${WARMUP}" --steps "${STEPS}" > "${RAW_DIR}/ws${ws}_mb${mb}.log" 2>&1
    [[ $? -eq 0 ]] && echo "ok" || echo "FAILED (see ${RAW_DIR}/ws${ws}_mb${mb}.log)"
  done
done

echo
echo "=== aggregating ==="

RAW_DIR="${RAW_DIR}" OUTDIR="${OUTDIR}" "${RUNNER[@]}" - <<'PYEOF'
import os, re, json, csv, glob
import numpy as np

raw, out = os.environ["RAW_DIR"], os.environ["OUTDIR"]

def extract_json(text):
    for s in (m.start() for m in re.finditer(r"\{", text)):
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[s:i+1])
                    except json.JSONDecodeError: break
    return None

def bus_factor(n): return 2.0*(n-1)/n

rows = []
for log in sorted(glob.glob(os.path.join(raw, "ws*_mb*.log"))):
    m = re.search(r"ws(\d+)_mb(\d+)\.log", os.path.basename(log))
    ws, mb = int(m.group(1)), int(m.group(2))
    txt = open(log).read()
    d = extract_json(txt)
    if d is None:
        rows.append(dict(world_size=ws, data_size_mb=mb, status="failed",
                         backend="", device="", avg_ms="", p50_ms="", p99_ms="",
                         max_ms="", std_ms="", algbw_GBps="", busbw_GBps=""))
        continue
    nbytes = mb*1024*1024
    avg = d["avg_ms"]
    algbw = nbytes/(avg/1e3)/1e9 if avg > 0 else 0.0
    busbw = algbw*bus_factor(ws)
    rows.append(dict(world_size=ws, data_size_mb=mb, status="ok",
                     backend=d.get("backend",""), device=d.get("device",""),
                     avg_ms=avg, p50_ms=d["p50_ms"], p99_ms=d["p99_ms"],
                     max_ms=d["max_ms"], std_ms=d["std_ms"],
                     algbw_GBps=round(algbw,2), busbw_GBps=round(busbw,2)))

cols = ["world_size","data_size_mb","status","backend","device",
        "avg_ms","p50_ms","p99_ms","max_ms","std_ms","algbw_GBps","busbw_GBps"]
csv_path = os.path.join(out, "summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in sorted(rows, key=lambda x:(x["world_size"], x["data_size_mb"])): w.writerow(r)
print("wrote", csv_path)

print("\n=== summary ===")
hdr = f"{'ws':>3} {'MB':>5} {'avg_ms':>9} {'p50':>8} {'p99':>8} {'busbw_GBps':>11} {'status':>8}"
print(hdr); print("-"*len(hdr))
for r in sorted(rows, key=lambda x:(x["world_size"], x["data_size_mb"])):
    print(f"{r['world_size']:>3} {r['data_size_mb']:>5} {str(r['avg_ms']):>9} "
          f"{str(r['p50_ms']):>8} {str(r['p99_ms']):>8} {str(r['busbw_GBps']):>11} {r['status']:>8}")

ok = [r for r in rows if r["status"]=="ok"]
if not ok:
    print("\n(no successful runs to plot)"); raise SystemExit
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except Exception as e:
    print(f"\n(skipping plots: {e})"); raise SystemExit

wss = sorted({r["world_size"] for r in ok})
sizes = sorted({r["data_size_mb"] for r in ok})
def series(ws, key):
    d = {r["data_size_mb"]: r[key] for r in ok if r["world_size"]==ws}
    return [d.get(s, np.nan) for s in sizes]

plt.figure(figsize=(7,5))
for ws in wss: plt.plot(sizes, series(ws,"avg_ms"), marker="o", label=f"{ws} ranks")
plt.xscale("log"); plt.yscale("log"); plt.grid(True, which="both", alpha=.3); plt.legend()
plt.xlabel("data size (MB)"); plt.ylabel("avg latency (ms)"); plt.title("All-reduce latency vs size")
p=os.path.join(out,"latency_vs_size.png"); plt.tight_layout(); plt.savefig(p,dpi=130); plt.close(); print("wrote",p)

plt.figure(figsize=(7,5))
for ws in wss: plt.plot(sizes, series(ws,"busbw_GBps"), marker="s", label=f"{ws} ranks")
plt.xscale("log"); plt.grid(True, which="both", alpha=.3); plt.legend()
plt.xlabel("data size (MB)"); plt.ylabel("bus bandwidth (GB/s)"); plt.title("All-reduce bus bandwidth vs size")
p=os.path.join(out,"busbw_vs_size.png"); plt.tight_layout(); plt.savefig(p,dpi=130); plt.close(); print("wrote",p)

mat = np.full((len(wss), len(sizes)), np.nan)
for i,ws in enumerate(wss):
    for j,s in enumerate(sizes):
        v=[r["avg_ms"] for r in ok if r["world_size"]==ws and r["data_size_mb"]==s]
        if v: mat[i,j]=v[0]
plt.figure(figsize=(7,4.5))
im=plt.imshow(mat, aspect="auto", cmap="viridis"); plt.colorbar(im, label="avg latency (ms)")
plt.xticks(range(len(sizes)), [f"{s}MB" for s in sizes]); plt.yticks(range(len(wss)), [str(w) for w in wss])
plt.xlabel("data size"); plt.ylabel("world size"); plt.title("Avg all-reduce latency (ms)")
for i in range(len(wss)):
    for j in range(len(sizes)):
        if not np.isnan(mat[i,j]): plt.text(j,i,f"{mat[i,j]:.1f}",ha="center",va="center",color="w",fontsize=8)
p=os.path.join(out,"heatmap_avg_ms.png"); plt.tight_layout(); plt.savefig(p,dpi=130); plt.close(); print("wrote",p)
PYEOF

echo
echo "done -> ${OUTDIR}/summary.csv + ${OUTDIR}/*.png"