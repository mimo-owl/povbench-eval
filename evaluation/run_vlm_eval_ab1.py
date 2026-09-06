"""
Ablation 1: All exploration images + red circle on images showing the observer furniture.

For each exploration image where the observer furniture is in_frame (visible),
draws a red filled circle at its projected 2D position.
The VLM still selects an image and estimates coordinates (standard A/B output).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_vlm_eval_ab0 import (
    run_eval as _run_eval_base,
    _load_exploration_images, _load_exploration_log_map,
    _get_pos, _project, _prepare_images as _prepare_images_ab0,
    SCREEN_WIDTH, SCREEN_HEIGHT,
)
from run_vlm_eval import _prompt_ab, merge_predictions

try:
    from PIL import Image, ImageDraw
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
    print("Warning: Pillow not available — annotation will be skipped.")

_OBSERVER_NOTE = (
    "In some images, a red circle marks the observer's furniture "
    "(the object they were working with at the moment they saw the target object). "
)


# ── annotation ────────────────────────────────────────────────────────────────

def _draw_red_circle(img_bytes: bytes, ux: int, vy: int, w: int, h: int) -> bytes:
    try:
        img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        r = max(8, int(min(w, h) * 0.022))
        draw.ellipse([ux - r, vy - r, ux + r, vy + r], fill="red", outline="white", width=2)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"    [ab1 annotate] {e}")
        return img_bytes


def _annotate_observer(img_bytes: bytes, obs_pos, camera: dict, w: int, h: int) -> bytes:
    """Draw red circle only if observer furniture is in_frame."""
    if not _PIL_OK or obs_pos is None or not camera:
        return img_bytes
    ux, vy, in_frame = _project(obs_pos, camera, w, h)
    if not in_frame or ux is None:
        return img_bytes
    return _draw_red_circle(img_bytes, ux, vy, w, h)


# ── image preparation ─────────────────────────────────────────────────────────

def _prepare_images(
    exp_images: list[tuple[str, Path]],
    entry: dict,
    exp_log_map: dict,
) -> list[bytes]:
    obs_id  = entry.get("object_observer", {}).get("objectId")
    obs_pos = _get_pos(entry.get("object_observer", {}))
    result = []
    for rel, img_path in exp_images:
        log_entry     = exp_log_map.get(rel, {})
        camera        = log_entry.get("camera", {})
        in_frame_hits = set(log_entry.get("object_in_frame_hits", []))
        w = int(camera.get("screen_width",  SCREEN_WIDTH))
        h = int(camera.get("screen_height", SCREEN_HEIGHT))
        img_bytes = img_path.read_bytes()
        # Only annotate if observer is actually rendered visible (not behind walls)
        if obs_id and obs_id in in_frame_hits and obs_pos and camera:
            ux, vy, _ = _project(obs_pos, camera, w, h)
            if ux is not None:
                img_bytes = _draw_red_circle(img_bytes, ux, vy, w, h)
        result.append(img_bytes)
    return result


# ── prompt ────────────────────────────────────────────────────────────────────

def _make_prompt(sentence: str, n: int, entry: dict, w: int, h: int) -> str:
    base = _prompt_ab(sentence, n, w, h)
    # Insert annotation note right after the image description sentence
    marker = "Each image is preceded by its label"
    idx = base.find(marker)
    if idx != -1:
        end = base.index("\n", idx) + 1
        return base[:end] + _OBSERVER_NOTE + "\n" + base[end:]
    return base


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
) -> None:
    _run_eval_base(
        dataset_dir=dataset_dir, output_dir=output_dir,
        vlm=vlm, types=types, max_pairs=max_pairs,
        worker_id=worker_id, num_workers=num_workers,
        qwen_model_dir=qwen_model_dir, ablation_tag="ab1",
        prepare_images_fn=_prepare_images,
        make_prompt_fn=_make_prompt,
    )


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
        description="Ablation 1: A/B with exploration images + red circle on observer position.")
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
