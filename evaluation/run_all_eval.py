"""
Batch runner for the clamped evaluation.

Runs eval_results.evaluate() over every saved prediction set under
eval_results/ WITHOUT re-running any VLM. For each run directory it writes
a per-run `eval_results.json` (symmetric clamp,
response-rate metric), then compiles a consolidated summary.

Output files are defined here (program-side) and are always written on run:
    eval_results/summary.csv   — one row per run + per-model means
    eval_results/summary.md    — human-readable table

Nothing else is modified: no VLM is re-run and the dataset is untouched.

See README.md for usage.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

# ── default paths (override on the CLI; see --help) ────────────────────────────
REPO                = Path(__file__).resolve().parents[1]
DEFAULT_PRED_ROOT   = REPO / "eval_results"
DEFAULT_DATASET_DIR = REPO / "dataset" / "POVBench"
# The consolidated summary is always written to <pred_root>/summary.{csv,md}.

# ── load eval_results.py as a module ──────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "eval_results", Path(__file__).parent / "eval_results.py"
)
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)


def _has_predictions(d: Path) -> bool:
    return any(
        p.name not in ("eval_results.json", "meta.json")
        for p in d.glob("predictions*.json")
    )


def _label(name: str) -> tuple[str, str]:
    """(model, run) from a run-dir name. 'POVBench_gemini25_0' -> ('gemini25','0')."""
    short = re.sub(r"^POVBench_", "", name)
    m = re.match(r"^(.*)_(\d+)$", short)
    if m:
        return m.group(1), m.group(2)
    return short, "-"


def _fmt(x, nd=4):
    return "" if x is None else f"{x:.{nd}f}"


def main(pred_root: Path, dataset_dir: Path) -> None:
    summary_csv = pred_root / "summary.csv"
    summary_md  = pred_root / "summary.md"

    run_dirs = sorted(
        d for d in pred_root.iterdir()
        if d.is_dir() and not d.name.endswith("_vs") and _has_predictions(d)
    )
    print(f"Found {len(run_dirs)} run directories with predictions.\n")

    rows: list[dict] = []
    for d in run_dirs:
        model, run = _label(d.name)
        print(f"──── {d.name} (model={model}, run={run}) ────")
        er.evaluate(
            predictions_path=d, dataset_dir=dataset_dir, output_dir=d,
            plot=False,
        )
        summary = json.loads((d / "eval_results.json").read_text())["summary"]
        ov = summary["overall"]
        pt = summary.get("per_type", {})
        rows.append({
            "run_dir":        d.name,
            "model":          model,
            "run":            run,
            "n_total":        summary["total"],
            "n_ok":           summary["n_ok"],
            "response_rate":  summary["response_rate"],
            "mean_l2":        ov.get("mean_l2"),
            "std_l2":         ov.get("std_l2"),
            "l2_a":           pt.get("a", {}).get("mean_l2"),
            "l2_b":           pt.get("b", {}).get("mean_l2"),
            "l2_c":           pt.get("c", {}).get("mean_l2"),
            "rr_a":           pt.get("a", {}).get("response_rate"),
            "rr_b":           pt.get("b", {}).get("response_rate"),
            "rr_c":           pt.get("c", {}).get("response_rate"),
        })
        print()

    # ── per-model means across runs ────────────────────────────────────────────
    import numpy as np
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    model_rows: list[dict] = []
    for model, rs in sorted(by_model.items()):
        def _mean(key):
            vals = [r[key] for r in rs if r[key] is not None]
            return float(np.mean(vals)) if vals else None
        model_rows.append({
            "model":          model,
            "n_runs":         len(rs),
            "mean_l2":        _mean("mean_l2"),
            "response_rate":  _mean("response_rate"),
            "l2_a":           _mean("l2_a"),
            "l2_b":           _mean("l2_b"),
            "l2_c":           _mean("l2_c"),
        })

    # ── write CSV ──────────────────────────────────────────────────────────────
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote per-run CSV → {summary_csv}")

    # ── write Markdown ─────────────────────────────────────────────────────────
    lines = []
    lines.append("# Evaluation summary\n")
    lines.append("## Per model (mean across runs)\n")
    lines.append("| Model | #runs | mean L2 | resp. rate | L2 A | L2 B | L2 C |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for m in model_rows:
        rr = "" if m["response_rate"] is None else f"{m['response_rate']*100:.1f}%"
        lines.append(f"| {m['model']} | {m['n_runs']} | {_fmt(m['mean_l2'])} | {rr} | "
                     f"{_fmt(m['l2_a'])} | {_fmt(m['l2_b'])} | {_fmt(m['l2_c'])} |")
    lines.append("\n## Per run\n")
    lines.append("| Run dir | n_ok/n_total | resp. rate | mean L2 | L2 A | L2 B | L2 C |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        rr = "" if r["response_rate"] is None else f"{r['response_rate']*100:.1f}%"
        lines.append(f"| {r['run_dir']} | {r['n_ok']}/{r['n_total']} | {rr} | "
                     f"{_fmt(r['mean_l2'])} | {_fmt(r['l2_a'])} | {_fmt(r['l2_b'])} | {_fmt(r['l2_c'])} |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Markdown summary → {summary_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch clamped evaluation over all saved prediction runs.")
    ap.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT,
                    help="Directory containing per-run prediction sub-directories "
                         f"(default: {DEFAULT_PRED_ROOT}).")
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                    help=f"Dataset directory with artifacts/ (default: {DEFAULT_DATASET_DIR}).")
    args = ap.parse_args()
    main(pred_root=args.pred_root, dataset_dir=args.dataset_dir)
