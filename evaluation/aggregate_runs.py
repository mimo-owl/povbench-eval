"""
Aggregate evaluation results across multiple runs and models into CSV files.

Reads eval_results.json files produced by eval_results.py and outputs:
  - all_runs_long.csv : one row per (model × run × item), all raw values

Each --run argument is: model_name:run_id:path/to/eval_dir
  model_name : arbitrary label (e.g. "gemini", "gpt", "gemma", "qwen")
  run_id     : integer run index (1, 2, 3, ...)
  path       : directory containing eval_results.json
Alternatively use --config to pass a JSON file:
  [{"model": "gemini", "run": 1, "eval_dir": "path/to/eval1"}, ...]

See README.md for usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# ── loaders ───────────────────────────────────────────────────────────────────

def _load_eval(eval_dir: Path) -> tuple[list[dict], dict, dict]:
    """Load eval_results.json. Returns (evaluations, summary, metadata)."""
    p = eval_dir / "eval_results.json"
    if not p.exists():
        raise FileNotFoundError(f"eval_results.json not found in {eval_dir}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("evaluations", []), data.get("summary", {}), data.get("metadata", {})


def _load_vlm_name(eval_dir: Path) -> str:
    """Try to read VLM key from prediction JSON files in eval_dir."""
    for pat in ("predictions.json", "predictions_0.json"):
        p = eval_dir / pat
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")).get("vlm", "")
            except Exception:
                pass
    for p in sorted(eval_dir.glob("predictions*.json")):
        try:
            v = json.loads(p.read_text(encoding="utf-8")).get("vlm", "")
            if v:
                return v
        except Exception:
            pass
    return ""


def _load_raw_response_map(eval_dir: Path) -> dict[tuple, str]:
    """
    Load raw_response from per-house prediction JSON files.
    Returns dict keyed by (house_id, pair_id, direction, type).
    """
    pred_map: dict[tuple, str] = {}
    for json_path in sorted(eval_dir.glob("train_house_*.json")):
        try:
            preds = json.loads(json_path.read_text(encoding="utf-8"))
            for p in preds:
                key = (p.get("house_id", ""), p["pair_id"], p["direction"], p["type"])
                pred_map[key] = p.get("raw_response", "")
        except Exception:
            pass
    return pred_map


# ── flatteners ────────────────────────────────────────────────────────────────

_LONG_FIELDS = [
    "model", "run_id", "eval_dir",
    "pair_id", "house_id", "direction", "type",
    "sentence",
    "status", "parse_error",
    "predicted_x", "predicted_y",
    "predicted_x_norm", "predicted_y_norm",
    "gt_x", "gt_y", "gt_in_frame",
    "error_x", "error_y", "error_l2",
    "error_l2_min", "error_l2_mean", "n_points",   # RoboPoint multi-point metrics
    "predicted_image", "camera_image",
    "raw_response",
]


def _row_from_eval(e: dict, model: str, run_id: int, eval_dir: str, raw_response: str = "") -> dict:
    gt2d = e.get("gt_position_2d") or {}
    return {
        "model":            model,
        "run_id":           run_id,
        "eval_dir":         eval_dir,
        "pair_id":          e.get("pair_id", ""),
        "house_id":         e.get("house_id", ""),
        "direction":        e.get("direction", ""),
        "type":             e.get("type", ""),
        "sentence":         e.get("sentence", ""),
        "status":           e.get("status", ""),
        "parse_error":      e.get("parse_error") or "",
        "predicted_x":      e.get("predicted_x", ""),
        "predicted_y":      e.get("predicted_y", ""),
        "predicted_x_norm": e.get("predicted_x_norm", ""),
        "predicted_y_norm": e.get("predicted_y_norm", ""),
        "gt_x":             gt2d.get("x", ""),
        "gt_y":             gt2d.get("y", ""),
        "gt_in_frame":      e.get("gt_in_frame", ""),
        "error_x":          e.get("error_x", ""),
        "error_y":          e.get("error_y", ""),
        "error_l2":         e.get("error_l2", ""),
        "error_l2_min":     e.get("error_l2_min", ""),
        "error_l2_mean":    e.get("error_l2_mean", ""),
        "n_points":         e.get("n_points", ""),
        "predicted_image":  e.get("predicted_image") or "",
        "camera_image":     e.get("camera_image") or "",
        "raw_response":     raw_response,
    }


# ── CSV writers ───────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        print(f"  Warning: no rows to write → {path}")
        return
    fields = fieldnames or list(rows[0].keys())
    # add any extra keys from later rows (e.g. per-run columns)
    extra = []
    for r in rows:
        for k in r:
            if k not in fields and k not in extra:
                extra.append(k)
    fields = fields + extra

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def aggregate(run_specs: list[tuple[str, int, Path]], output_dir: Path) -> None:
    """
    run_specs: list of (model_name, run_id, eval_dir_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict] = []

    for model, run_id, eval_dir in sorted(run_specs, key=lambda x: (x[0], x[1])):
        print(f"\nLoading: model={model!r} run={run_id} dir={eval_dir}")
        try:
            evals, summary, metadata = _load_eval(eval_dir)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
            continue

        vlm_key = metadata.get("vlm", "") or _load_vlm_name(eval_dir)
        print(f"  vlm_key={vlm_key!r}  evaluations={len(evals)}")
        ok_count = sum(1 for e in evals if e.get("status") == "ok")
        print(f"  ok={ok_count}  total={len(evals)}")

        raw_map = _load_raw_response_map(eval_dir)
        print(f"  raw_response map: {len(raw_map)} entries")

        for e in evals:
            key = (e.get("house_id", ""), e.get("pair_id", ""), e.get("direction", ""), e.get("type", ""))
            raw = raw_map.get(key, "")
            long_rows.append(_row_from_eval(e, model, run_id, str(eval_dir), raw_response=raw))

    print(f"\nTotal rows (all models × runs): {len(long_rows)}")

    _write_csv(output_dir / "all_runs_long.csv", long_rows, fieldnames=_LONG_FIELDS)

    print(f"\nDone. Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate VLM evaluation results across runs/models into CSV files."
    )
    parser.add_argument(
        "--run", dest="runs", action="append", metavar="MODEL:RUN_ID:EVAL_DIR",
        help="One run spec: model_name:run_id:path/to/eval_dir. Repeat for each run.",
    )
    parser.add_argument(
        "--config", default=None,
        help='JSON file with list of {"model":..., "run":..., "eval_dir":...}',
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to write CSV files (default: results/)",
    )
    args = parser.parse_args()

    run_specs: list[tuple[str, int, Path]] = []

    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for entry in cfg:
            run_specs.append((entry["model"], int(entry["run"]), Path(entry["eval_dir"])))

    for spec in (args.runs or []):
        parts = spec.split(":", 2)
        if len(parts) != 3:
            print(f"ERROR: invalid --run spec {spec!r} (expected MODEL:RUN_ID:PATH)", file=sys.stderr)
            sys.exit(1)
        model, run_id_str, path = parts
        run_specs.append((model, int(run_id_str), Path(path)))

    if not run_specs:
        parser.print_help()
        sys.exit(1)

    aggregate(run_specs, Path(args.output_dir))


if __name__ == "__main__":
    main()
