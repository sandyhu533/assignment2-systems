#!/usr/bin/env python
"""
Sweep benchmark.py over (mode, size, warmup) and emit a results table
in markdown / latex / typst.

Usage:
    uv run cs336_systems/sweep.py
    uv run cs336_systems/sweep.py --formats latex typst --out results/
    uv run cs336_systems/sweep.py --modes forward forward_backward --sizes small medium
    uv run cs336_systems/sweep.py --extra -- --batch-size 2 --seq-len 256
"""
import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODES   = ["forward", "forward_backward", "full"]
SIZES   = ["small", "medium", "large", "xl", "10B"]
WARMUPS = [0, 1, 5]
STEPS   = 10
RESULT_RE = re.compile(r"^RESULT_JSON\s+(\{.*\})\s*$", re.MULTILINE)


def _print_summary(r: dict) -> None:
    """Pretty one-line summary printed after each successful run."""
    def g(k, fmt="{:.2f}", default="—"):
        v = r.get(k)
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            return default
        try:
            return fmt.format(v)
        except (TypeError, ValueError):
            return str(v)

    print(
        f"  ✓ {g('mean_ms')} ± {g('std_ms')} ms"
        f"  (median {g('median_ms')}, min {g('min_ms')}, p95 {g('p95_ms')})"
        f"  | mem {g('peak_reserve_mem_gb')} GB"
        f"  | mem {g('peak_alloc_mem_gb')} GB"
        f"  | {g('tokens_per_sec', '{:.0f}')} tok/s"
        f"  | {g('tflops')} TFLOPS"
        f"  | MFU {g('mfu', '{:.1%}')}",
        file=sys.stderr,
    )


def run_one(mode: str, size: str, warmup: int, steps: int, extra: list[str]) -> dict | None:
    cmd = [
        "uv", "run", "cs336_systems/benchmark.py",
        "--warmup", str(warmup),
        "--steps",  str(steps),
        "--mode",   mode,
        "--size",   size,
        *extra,
    ]
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"  ⚠ timeout", file=sys.stderr)
        return {"mode": mode, "size": size, "warmup": warmup,
                "steps": steps, "error": "timeout"}

    if proc.returncode != 0:
        # Common: OOM on xl/10B with small GPUs — record and continue.
        err_tail = proc.stderr.strip().splitlines()[-3:]
        print(f"  ⚠ rc={proc.returncode}: {err_tail}", file=sys.stderr)
        return {"mode": mode, "size": size, "warmup": warmup,
                "steps": steps, "error": f"rc={proc.returncode}"}

    m = RESULT_RE.search(proc.stdout)
    if not m:
        print(f"  ⚠ no RESULT_JSON line found in stdout", file=sys.stderr)
        return {"mode": mode, "size": size, "warmup": warmup,
                "steps": steps, "error": "no_result"}

    result = json.loads(m.group(1))
    _print_summary(result)
    return result


def build_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # Preserve our enumeration order (not alphabetical).
    df["mode"]   = pd.Categorical(df["mode"],   categories=MODES,   ordered=True)
    df["size"]   = pd.Categorical(df["size"],   categories=SIZES,   ordered=True)
    df["warmup"] = pd.Categorical(df["warmup"], categories=WARMUPS, ordered=True)
    df = df.sort_values(["mode", "size", "warmup"]).reset_index(drop=True)
    return df


def to_typst(df: pd.DataFrame, value_col: str = "mean_ms", err_col: str | None = "std_ms") -> str:
    """
    Pivot to a (size × warmup) table per mode, formatted for Typst.
    Output uses Typst's #table(...) syntax.
    """
    chunks = []
    for mode in MODES:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue

        def fmt(row):
            if pd.isna(row.get(value_col)):
                return f"err:{row.get('error', '?')}"
            if err_col and not pd.isna(row.get(err_col)):
                return f"{row[value_col]:.2f} ± {row[err_col]:.2f}"
            return f"{row[value_col]:.2f}"

        sub = sub.copy()
        sub["cell"] = sub.apply(fmt, axis=1)
        pivot = sub.pivot(index="size", columns="warmup", values="cell")
        pivot = pivot.reindex(index=SIZES, columns=WARMUPS)

        n_cols = len(pivot.columns) + 1
        header = ["size"] + [f"warmup={w}" for w in pivot.columns]
        lines = [
            f"// mode = {mode}",
            f"#figure(",
            f"  table(",
            f"    columns: {n_cols},",
            f"    align: (left, " + ", ".join(["right"] * (n_cols - 1)) + "),",
            f"    table.header(" + ", ".join(f"[*{h}*]" for h in header) + "),",
        ]
        for size, row in pivot.iterrows():
            cells = [f"[{size}]"] + [f"[{(v if pd.notna(v) else '—')}]" for v in row]
            lines.append("    " + ", ".join(cells) + ",")
        lines += [
            f"  ),",
            f'  caption: [Benchmark results, mode = `{mode}`, '
            f'values in ms (mean ± std over {STEPS} steps)],',
            f")",
            "",
        ]
        chunks.append("\n".join(lines))
    return "\n".join(chunks)


def to_latex(df: pd.DataFrame, value_col: str = "mean_ms", err_col: str | None = "std_ms") -> str:
    chunks = []
    for mode in MODES:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue

        def fmt(row):
            if pd.isna(row.get(value_col)):
                return f"err:{row.get('error', '?')}"
            if err_col and not pd.isna(row.get(err_col)):
                return f"{row[value_col]:.2f} $\\pm$ {row[err_col]:.2f}"
            return f"{row[value_col]:.2f}"

        sub = sub.copy()
        sub["cell"] = sub.apply(fmt, axis=1)
        pivot = sub.pivot(index="size", columns="warmup", values="cell")
        pivot = pivot.reindex(index=SIZES, columns=WARMUPS)

        tex = pivot.to_latex(
            na_rep="---",
            caption=f"Benchmark results, mode = \\texttt{{{mode}}}, "
                    f"values in ms (mean $\\pm$ std over {STEPS} steps).",
            label=f"tab:bench-{mode}",
            escape=False,
        )
        chunks.append(tex)
    return "\n".join(chunks)


def to_markdown(df: pd.DataFrame, value_col: str = "mean_ms", err_col: str | None = "std_ms") -> str:
    chunks = []
    for mode in MODES:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue

        def fmt(row):
            if pd.isna(row.get(value_col)):
                return f"err:{row.get('error', '?')}"
            if err_col and not pd.isna(row.get(err_col)):
                return f"{row[value_col]:.2f} ± {row[err_col]:.2f}"
            return f"{row[value_col]:.2f}"

        sub = sub.copy()
        sub["cell"] = sub.apply(fmt, axis=1)
        pivot = sub.pivot(index="size", columns="warmup", values="cell")
        pivot = pivot.reindex(index=SIZES, columns=WARMUPS)
        pivot.columns = [f"warmup={w}" for w in pivot.columns]

        chunks.append(f"### mode = `{mode}`  (ms, mean ± std over {STEPS} steps)\n")
        chunks.append(pivot.to_markdown())
        chunks.append("")
    return "\n".join(chunks)


def print_final_summary(df: pd.DataFrame) -> None:
    """Sweep-level recap printed at the end."""
    has_mean = "mean_ms" in df.columns
    ok   = df.dropna(subset=["mean_ms"]) if has_mean else df.iloc[0:0]
    fail = df[df["mean_ms"].isna()]      if has_mean else df

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Sweep complete: {len(ok)} ok, {len(fail)} failed", file=sys.stderr)

    if len(ok) > 0 and "mfu" in ok.columns and ok["mfu"].notna().any():
        best = ok.loc[ok["mfu"].idxmax()]
        print(f"Best MFU : {best['mfu']:.1%}  "
              f"({best['mode']}/{best['size']}/warmup={best['warmup']})",
              file=sys.stderr)

    if len(fail) > 0:
        print("Failed runs:", file=sys.stderr)
        for _, row in fail.iterrows():
            print(f"  - {row['mode']}/{row['size']}/warmup={row['warmup']}: "
                  f"{row.get('error', 'unknown')}", file=sys.stderr)

    print(f"{'='*60}\n", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes",   nargs="+", default=MODES,   choices=MODES)
    ap.add_argument("--sizes",   nargs="+", default=SIZES,   choices=SIZES)
    ap.add_argument("--warmups", nargs="+", default=WARMUPS, type=int)
    ap.add_argument("--steps",   type=int,  default=STEPS)
    ap.add_argument("--formats", nargs="+", default=["csv"],
                    choices=["markdown", "latex", "typst", "csv"])
    ap.add_argument("--out",     type=Path, default=Path("results"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print combinations without running.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra args forwarded to benchmark.py after `--`.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    combos = list(itertools.product(args.modes, args.sizes, args.warmups))
    print(f"Total runs: {len(combos)}", file=sys.stderr)

    if args.dry_run:
        for c in combos:
            print(c)
        return

    rows = []
    for i, (mode, size, warmup) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] mode={mode} size={size} warmup={warmup}",
              file=sys.stderr)
        rows.append(run_one(mode, size, warmup, args.steps, args.extra))

    df = build_dataframe(rows)

    # Always dump raw CSV for re-rendering later without re-running.
    csv_path = args.out / "raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw → {csv_path}", file=sys.stderr)

    if "markdown" in args.formats:
        (args.out / "table.md").write_text(to_markdown(df))
        print(f"Saved md  → {args.out/'table.md'}", file=sys.stderr)
    if "latex" in args.formats:
        (args.out / "table.tex").write_text(to_latex(df))
        print(f"Saved tex → {args.out/'table.tex'}", file=sys.stderr)
    if "typst" in args.formats:
        (args.out / "table.typ").write_text(to_typst(df))
        print(f"Saved typ → {args.out/'table.typ'}", file=sys.stderr)

    print_final_summary(df)

    # Echo the markdown table to stdout for quick eyeballing.
    print("\n" + to_markdown(df))


if __name__ == "__main__":
    main()