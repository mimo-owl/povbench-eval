"""
Ablation 2: ab1 + blue bounding box on images showing the anchor (landmark) object.

Builds on ab1: the observer red circle is drawn on images where observer is in_frame.
Additionally, a blue rectangle is drawn on images where the anchor object is in_frame,
using its axisAlignedBoundingBox corners. Falls back to a blue circle if corners unavailable.
The VLM still selects an image and estimates coordinates (standard A/B output).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_vlm_eval_ab1 import (
    _draw_red_circle, _OBSERVER_NOTE,
)
from run_vlm_eval_ab0 import (
    run_eval as _run_eval_base,
    _get_pos, _project,
    SCREEN_WIDTH, SCREEN_HEIGHT,
)
from run_vlm_eval import _prompt_ab, merge_predictions

try:
    from PIL import Image, ImageDraw
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_ANCHOR_NOTE = (
    "a blue rectangle marks the anchor object mentioned in the sentence "
    "(the landmark used to describe where the target object is located). "
)


# ── annotation ────────────────────────────────────────────────────────────────

def _annotate_anchor(img_bytes: bytes, lm_meta: dict, camera: dict, w: int, h: int) -> bytes:
    """Draw blue bbox (or circle) around anchor object if in_frame."""
    if not _PIL_OK or not camera:
        return img_bytes
    try:
        lm_raw  = lm_meta.get("raw_metadata", {})
        corners = lm_raw.get("axisAlignedBoundingBox", {}).get("cornerPoints", [])

        img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        if corners:
            us, vs = [], []
            for corner in corners:
                ux, vy, _ = _project(corner, camera, w, h)
                if ux is not None:
                    us.append(ux)
                    vs.append(vy)
            if us and vs:
                x0 = max(0, min(us));  y0 = max(0, min(vs))
                x1 = min(w - 1, max(us)); y1 = min(h - 1, max(vs))
                if x1 > x0 and y1 > y0:
                    draw.rectangle([x0, y0, x1, y1], outline="blue", width=3)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    return buf.getvalue()
        else:
            pos = _get_pos(lm_meta)
            if pos is not None:
                ux, vy, in_frame = _project(pos, camera, w, h)
                if in_frame and ux is not None:
                    r = max(15, int(min(w, h) * 0.04))
                    draw.ellipse([ux - r, vy - r, ux + r, vy + r],
                                 outline="blue", width=3)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    return buf.getvalue()
    except Exception as e:
        print(f"    [ab2 annotate] {e}")
    return img_bytes


# ── image preparation ─────────────────────────────────────────────────────────

def _prepare_images(
    exp_images: list[tuple[str, Path]],
    entry: dict,
    exp_log_map: dict,
) -> list[bytes]:
    obs_id  = entry.get("object_observer", {}).get("objectId")
    obs_pos = _get_pos(entry.get("object_observer", {}))
    lm_id   = entry.get("object_landmark",  {}).get("objectId")
    lm_meta = entry.get("object_landmark",  {})
    result = []
    for rel, img_path in exp_images:
        log_entry     = exp_log_map.get(rel, {})
        camera        = log_entry.get("camera", {})
        in_frame_hits = set(log_entry.get("object_in_frame_hits", []))
        w = int(camera.get("screen_width",  SCREEN_WIDTH))
        h = int(camera.get("screen_height", SCREEN_HEIGHT))
        img_bytes = img_path.read_bytes()
        # Observer: red circle only if actually rendered visible
        if obs_id and obs_id in in_frame_hits and obs_pos and camera:
            ux, vy, _ = _project(obs_pos, camera, w, h)
            if ux is not None:
                img_bytes = _draw_red_circle(img_bytes, ux, vy, w, h)
        # Anchor: blue bbox only if actually rendered visible
        if lm_id and lm_id in in_frame_hits:
            img_bytes = _annotate_anchor(img_bytes, lm_meta, camera, w, h)
        result.append(img_bytes)
    return result


# ── prompt ────────────────────────────────────────────────────────────────────

def _make_prompt(sentence: str, n: int, entry: dict, w: int, h: int) -> str:
    from run_vlm_eval_ab1 import _OBSERVER_NOTE
    base = _prompt_ab(sentence, n, w, h)
    note = _OBSERVER_NOTE + "In some images, " + _ANCHOR_NOTE
    marker = "Each image is preceded by its label"
    idx = base.find(marker)
    if idx != -1:
        end = base.index("\n", idx) + 1
        return base[:end] + note + "\n" + base[end:]
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
        qwen_model_dir=qwen_model_dir, ablation_tag="ab2",
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
        description="Ablation 2: A/B with exploration images + observer circle + anchor bbox.")
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
