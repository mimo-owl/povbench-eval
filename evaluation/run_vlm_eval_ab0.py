"""
Ablation 0: Type A/B with all exploration images, no annotation.

Baseline for the ablation series. Passes all exploration images (up to 30) to the
VLM with the standard A/B prompt (select image + estimate coordinates).
Identical in structure to the standard A/B evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_vlm_eval import (
    _call_vlm, _parse_ab_response, _prompt_ab,
    _load_dataset, merge_predictions,
    SCREEN_WIDTH, SCREEN_HEIGHT, QWEN_DEFAULT_MODEL_DIR,
)


MAX_EXPLORATION_IMAGES = 30


# ── shared utilities (imported by ab1, ab2, ab3) ──────────────────────────────

def _load_exploration_images(
    artifacts_dir: Path,
    house_id: str,
    max_images: int = MAX_EXPLORATION_IMAGES,
) -> list[tuple[str, Path]]:
    """Return [(rel_path, abs_path), ...] sorted by name, up to max_images."""
    exp_dir = artifacts_dir / house_id / "exploration_images"
    if not exp_dir.exists():
        return []
    paths = sorted(p for p in exp_dir.glob("*.png") if "depth" not in p.name)
    return [(f"exploration_images/{p.name}", p) for p in paths[:max_images]]


def _load_exploration_log_map(artifacts_dir: Path, house_id: str) -> dict[str, dict]:
    """Return {image_rel_path: log_entry} (log_entry contains 'camera' dict)."""
    p = artifacts_dir / house_id / "exploration_log.json"
    if not p.exists():
        return {}
    log = json.loads(p.read_text(encoding="utf-8"))
    return {e["image_path"]: e for e in log if "image_path" in e}


def _get_pos(obj_meta: dict):
    """Extract 3D center position from object metadata dict."""
    raw = obj_meta.get("raw_metadata", {})
    pos = raw.get("position")
    if pos:
        return pos
    return raw.get("axisAlignedBoundingBox", {}).get("center")


def _project(pos, camera: dict, w: int, h: int):
    """Project 3D world pos → (px, py, in_frame). Returns (None, None, False) on failure."""
    try:
        if isinstance(pos, dict):
            tx, ty, tz = pos["x"], pos["y"], pos["z"]
        else:
            tx, ty, tz = float(pos[0]), float(pos[1]), float(pos[2])

        cp = camera.get("cam_pos", [0, 0, 0])
        cx, cy, cz = float(cp[0]), float(cp[1]), float(cp[2])
        yaw = math.radians(camera.get("cam_yaw", 0))
        hor = math.radians(camera.get("cam_horizon", 0))
        fov = camera.get("fov", 100.0)

        dx, dy, dz = tx - cx, ty - cy, tz - cz
        right   = ( math.cos(yaw),                         0.0,          -math.sin(yaw))
        forward = ( math.sin(yaw) * math.cos(hor),        -math.sin(hor), math.cos(yaw) * math.cos(hor))
        up      = ( math.sin(yaw) * math.sin(hor),         math.cos(hor), math.cos(yaw) * math.sin(hor))

        x_cam = dx*right[0]   + dy*right[1]   + dz*right[2]
        y_cam = dx*up[0]      + dy*up[1]       + dz*up[2]
        z_cam = dx*forward[0] + dy*forward[1]  + dz*forward[2]

        if z_cam <= 1e-6:
            return None, None, False

        f = (w / 2.0) / math.tan(math.radians(fov / 2.0))
        u = f * x_cam / z_cam + w / 2.0
        v = h / 2.0 - f * y_cam / z_cam
        in_frame = (0 <= u <= w - 1) and (0 <= v <= h - 1)
        return int(u), int(v), in_frame
    except Exception:
        return None, None, False


# ── image bytes (overridden in ab1 / ab2) ─────────────────────────────────────

def _prepare_images(
    exp_images: list[tuple[str, Path]],
    entry: dict,
    exp_log_map: dict,
) -> list[bytes]:
    """Return raw bytes for each exploration image (no annotation in ab0)."""
    return [p.read_bytes() for _, p in exp_images]


# ── prompt (overridden in ab3) ────────────────────────────────────────────────

def _make_prompt(sentence: str, n: int, entry: dict, w: int, h: int) -> str:
    return _prompt_ab(sentence, n, w, h)


# ── eval loop ─────────────────────────────────────────────────────────────────

def run_eval(
    dataset_dir: Path,
    output_dir: Path,
    vlm: str = "gemini",
    types=None,
    max_pairs=None,
    worker_id: int = 0,
    num_workers: int = 1,
    qwen_model_dir=None,
    ablation_tag: str = "ab0",
    prepare_images_fn=None,
    make_prompt_fn=None,
    max_new_tokens: int = 256,
) -> None:
    if prepare_images_fn is None:
        prepare_images_fn = _prepare_images
    if make_prompt_fn is None:
        make_prompt_fn = _make_prompt

    if types is None:
        types = ["a", "b"]
    types_set = set(t.lower() for t in types) & {"a", "b"}

    artifacts_dir = dataset_dir / "artifacts"
    all_entries = _load_dataset(dataset_dir)
    entries = [e for i, e in enumerate(all_entries) if i % num_workers == worker_id]
    print(f"Loaded {len(all_entries)} pairs | worker {worker_id}: {len(entries)} entries")

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = "predictions.json" if num_workers == 1 else f"predictions_{worker_id}.json"
    output_path = output_dir / fname

    predictions: list[dict] = []
    done_keys: set[tuple] = set()
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            predictions = existing.get("predictions", [])
            done_keys = {
                (p.get("house_id", ""), p["pair_id"], p["direction"], p["type"])
                for p in predictions
            }
            print(f"Resuming: {len(predictions)} saved.")
        except Exception as exc:
            print(f"  Warning: {exc}")

    client_state: dict = {"worker_id": worker_id, "model_dir": qwen_model_dir or QWEN_DEFAULT_MODEL_DIR,
                          "max_new_tokens": max_new_tokens}
    run_start = time.time()
    cmd_count = 0
    total_tokens = 0
    log_path = output_dir / f"timing_{worker_id}.log"

    def _log(msg):
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    metadata_base = {
        "dataset_dir": str(dataset_dir), "vlm": vlm,
        "types": sorted(types_set), "ablation": ablation_tag,
        "screen_width": SCREEN_WIDTH, "screen_height": SCREEN_HEIGHT,
        "worker_id": worker_id, "num_workers": num_workers,
    }

    def _flush():
        output_path.write_text(
            json.dumps({**metadata_base,
                        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "n_predictions": len(predictions), "predictions": predictions},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    processed = 0
    for entry in entries:
        if max_pairs is not None and processed >= max_pairs:
            break

        pair_id  = entry.get("pair_id", "?")
        house_id = entry.get("house_id", "")
        stage5   = entry.get("stage5", {})
        camera   = entry.get("camera", {})

        exp_images  = _load_exploration_images(artifacts_dir, house_id)
        exp_log_map = _load_exploration_log_map(artifacts_dir, house_id)
        if not exp_images:
            print(f"  [{pair_id}] No exploration images — skipping")
            continue

        candidate_rels = [r for r, _ in exp_images]
        c_w = int(camera.get("screen_width",  SCREEN_WIDTH))
        c_h = int(camera.get("screen_height", SCREEN_HEIGHT))

        for direction in sorted(stage5.keys()):
            s5 = stage5.get(direction, {})
            for sent_type, key_s in [("a", "type_a"), ("b", "type_b")]:
                if sent_type not in types_set:
                    continue
                sentence = s5.get(key_s, "")
                if not sentence or (house_id, pair_id, direction, sent_type) in done_keys:
                    continue

                images_bytes = prepare_images_fn(exp_images, entry, exp_log_map)
                n = len(images_bytes)
                labels = [f"Image {i}:" for i in range(n)]
                prompt = make_prompt_fn(sentence, n, entry, c_w, c_h)

                print(f"  [{pair_id}/{direction}/{sent_type.upper()}] {n} images")
                raw, parse_error, parsed = "", None, None
                try:
                    raw = _call_vlm(vlm, images_bytes, labels, prompt, client_state)
                    print(f"    Response: {raw[:120]}")
                    parsed = _parse_ab_response(raw, c_w, c_h)
                    if parsed is None:
                        parse_error = "failed to parse JSON"
                except Exception as exc:
                    parse_error = str(exc)
                    print(f"    Error: {exc}")

                tokens = client_state.pop("last_tokens", 0)
                total_tokens += tokens
                sel_idx = parsed["selected_image"] if parsed else None
                predicted_image = (
                    candidate_rels[sel_idx]
                    if sel_idx is not None and 0 <= sel_idx < len(candidate_rels)
                    else None
                )
                predictions.append({
                    "pair_id": pair_id, "house_id": house_id,
                    "direction": direction, "type": sent_type,
                    "sentence": sentence,
                    "candidate_images": candidate_rels,
                    "predicted_image":  predicted_image,
                    "predicted_x": parsed["x"] if parsed else None,
                    "predicted_y": parsed["y"] if parsed else None,
                    "raw_response": raw, "parse_error": parse_error, "tokens": tokens,
                })
                _flush()
                processed += 1
                cmd_count += 1

    _flush()
    elapsed = time.time() - run_start
    mins, secs = divmod(int(elapsed), 60)
    _log(f"\n── Worker {worker_id} summary ({ablation_tag}) ──────────────────────\n"
         f"  Commands  : {cmd_count}\n  Elapsed   : {mins}m {secs:02d}s\n"
         f"  Throughput: {cmd_count/elapsed*60 if elapsed else 0:.1f} cmd/min\n"
         f"  Tokens    : {total_tokens:,}\n  Output    : {output_path}\n"
         f"────────────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from dotenv import load_dotenv
        _env = Path(__file__).resolve().parents[1] / ".env"
        if _env.exists():
            load_dotenv(dotenv_path=_env)
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Ablation 0: A/B with all exploration images, no annotation (baseline).")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--vlm", default="gemini",
                        choices=["gemma", "gemini", "gemini-robotics", "gpt", "qwen"])
    parser.add_argument("--qwen-model-dir", default=None)
    parser.add_argument("--types", nargs="+", default=["a", "b"], choices=["a", "b"])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--worker-id",   type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.merge:
        merge_predictions(output_dir, args.num_workers)
        return

    run_eval(
        dataset_dir=Path(args.dataset_dir), output_dir=output_dir,
        vlm=args.vlm, types=args.types, max_pairs=args.max_pairs,
        worker_id=args.worker_id, num_workers=args.num_workers,
        qwen_model_dir=args.qwen_model_dir,
    )


if __name__ == "__main__":
    main()
