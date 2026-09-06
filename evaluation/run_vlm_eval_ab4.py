"""
Ablation 4: Chain-of-thought prompting (variant) with all exploration images.

Same structure as ab3. Edit _make_prompt below to implement a different CoT strategy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_vlm_eval_ab0 import (
    run_eval as _run_eval_base,
    _prepare_images,
)
from run_vlm_eval import _coord_description, merge_predictions


# ── prompt ────────────────────────────────────────────────────────────────────

def _make_prompt(sentence: str, n: int, _entry: dict, w: int, h: int) -> str:
    return (
        f"You are the intelligent brain system of a home-assistance robot. "
        f"You are given first-person perspective images captured during the robot's exploration of a home.\n\n"
        f"The {n} images above (labeled Image 0 through Image {n - 1}) were all taken "
        f"during exploration of the same single-story house. "
        f"Each image is preceded by its label (\"Image 0:\", \"Image 1:\", etc.).\n\n"
        f"Sentence: \"{sentence}\"\n\n"
        f"The resident has asked the robot to retrieve an object and is describing where they last saw it. "
        f"Your mission is to predict, as accurately as possible, where that object is located, "
        f"based on the resident's description in the sentence above.\n\n"
        f"Think step by step:\n\n"
        f"Step 1 — Image selection: Among the {n} images, identify the one that best represents "
        f"the scene described in the sentence — the image where both the furniture the observer "
        f"was working with and the spatial landmark mentioned in the sentence are most clearly visible. "
        f"Note its index.\n\n"
        f"Step 2 — Observer position: In the selected image, identify the furniture the observer "
        f"was working with at the time. Note roughly where it appears and what direction "
        f"the observer would be facing.\n\n"
        f"Step 3 — Anchor landmark: Identify the spatial landmark mentioned in the sentence "
        f"in the selected image. \n\n"
        f"Step 4 — Spatial reasoning:\n"
        f"  4a) Directional frame: Define the viewing direction as from the observer's furniture (Step 2) "
        f"toward the anchor object (Step 3). "
        f"Use the following spatial relations, all from the observer's perspective when facing the anchor object:\n"
        f"      - LEFT / RIGHT: the object is to the left or right side of the anchor object\n"
        f"      - FRONT: the object is closer to the observer than the anchor object "
        f"      - ON TOP: the object is physically resting on or above the anchor object "
        f"      - OTHER: the spatial relation does not clearly fit any of the above "
        f"(e.g., behind the anchor, diagonally distant)\n\n"
        f"  4b) Anchor coordinate: Estimate the pixel coordinate (x, y) of the anchor object in the image.\n\n"
        f"  4c) Scene inventory: Identify two or three objects that are visible in the selected image "
        f"and closest to the anchor object, excluding the observer's furniture. For each object, state:\n"
        f"      - Its name or brief description\n"
        f"      - Its spatial relation to the anchor object (LEFT, RIGHT, FRONT, ON TOP, or OTHER)\n"
        f"      - Its estimated pixel coordinate (x, y) in the image\n\n"
        f"  4d) Side identification: Re-read the sentence and determine the spatial relation of the target object "
        f"to the anchor object (LEFT, RIGHT, FRONT, or ON TOP).\n\n"
        f"  4e) Target location estimate: Using the anchor coordinate from 4b and the surrounding "
        f"objects' coordinates from 4c as spatial reference points, estimate the most plausible pixel "
        f"coordinate for the target object. Place it according to the spatial relation identified in 4d "
        f"— LEFT, RIGHT, FRONT, or ON TOP relative to the anchor — and use the relative "
        f"spacing and scale of the inventoried objects to judge and make as accurate a position estimate "
        f"as possible.\n\n"
        f"Step 5 — Conclusion: State your selected image index and final coordinate estimate.\n\n"
        f"{_coord_description(w, h)}\n\n"
        f"IMPORTANT: When predicting coordinates, pay close attention to the origin position and "
        f"axis directions, and ensure all values fall within the valid range.\n\n"
        f"Write your step-by-step reasoning freely. "
        f"End your response with ONLY the final JSON on the last line (no other text after it):\n"
        f'{{\"selected_image\": <integer 0-{n - 1}>, \"x\": <integer 0-{w - 1}>, \"y\": <integer 0-{h - 1}>}}'
    )


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
        qwen_model_dir=qwen_model_dir, ablation_tag="ab4",
        prepare_images_fn=_prepare_images,
        make_prompt_fn=_make_prompt,
        max_new_tokens=1024,
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
        description="Ablation 4: A/B with exploration images + chain-of-thought prompt (variant).")
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
