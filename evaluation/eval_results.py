"""
VLM Spatial-Reasoning Evaluation — Metrics and Plots

Pure offline recomputation against saved predictions (no VLM is re-run, dataset
untouched). Loads predictions from run_vlm_eval.py, computes localization error
against ground-truth 3D positions (projected to 2D), and produces summary plots.

Metric definition:
  1. GT heights are corrected from the stage-3 surface verdicts, matching the
     paper's numbers (Appendix Table 7 / gt_surface_corrected=True).
  2. Symmetric clamping: both predicted and GT normalized coords are clamped to
     [0, 1] before L2.
  3. Response rate (n_ok / n_total) is reported next to L2 for every group
     (overall, per type, per direction); L2 is averaged over answered instances only.

Coordinate system:
    AI2THOR uses Unity's left-handed world space: X=right, Y=up, Z=forward.
    Camera orientation is given by cam_yaw (Y-rotation, degrees) and cam_horizon
    (X-rotation = pitch, degrees; positive = looking down). Projected 2D coordinates
    are normalized to [0, 1]: (0, 0) = top-left, (1, 1) = bottom-right.

See README.md for usage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# ── geometry ──────────────────────────────────────────────────────────────────

def _camera_frame(cam_yaw_deg: float, cam_horizon_deg: float) -> tuple[tuple, tuple, tuple]:
    """
    Compute (right, up, forward) unit vectors from AI2THOR camera angles.

    cam_yaw_deg   : rotation around Y axis (Unity convention, degrees)
    cam_horizon_deg: pitch, positive = camera tilts down (degrees)

    Derivation (verified against evidence.observer_frame in dataset):
      yaw rotation maps Z=(0,0,1) → forward_flat=(sin y, 0, cos y)
      horizon rotation tilts forward_flat down by horizon around the right axis
    """
    yaw = math.radians(cam_yaw_deg)
    hor = math.radians(cam_horizon_deg)

    right   = (math.cos(yaw),                          0.0,           -math.sin(yaw))
    forward = (math.sin(yaw) * math.cos(hor),          -math.sin(hor), math.cos(yaw) * math.cos(hor))
    up      = (math.sin(yaw) * math.sin(hor),           math.cos(hor), math.cos(yaw) * math.sin(hor))
    return right, up, forward


def project_3d_to_2d(
    target_pos: dict | list | tuple,
    cam_pos:    list | tuple,
    cam_yaw_deg:    float,
    cam_horizon_deg: float,
    fov_deg:      float = 100.0,
    screen_width:  int  = 512,
    screen_height: int  = 512,
) -> tuple[float, float, bool]:
    """
    Project a 3D world point to normalized image coordinates.

    Returns:
       (x_norm, y_norm, in_frame)
        x_norm, y_norm ∈ [0, 1] (clamped); in_frame=False if target is behind camera
        or if the projected point falls outside the image boundaries before clamping.
    """
    if isinstance(target_pos, dict):
        tx, ty, tz = target_pos["x"], target_pos["y"], target_pos["z"]
    else:
        tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])

    cx, cy, cz = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])

    dx, dy, dz = tx - cx, ty - cy, tz - cz

    right, up, forward = _camera_frame(cam_yaw_deg, cam_horizon_deg)

    x_cam = dx * right[0]   + dy * right[1]   + dz * right[2]
    y_cam = dx * up[0]      + dy * up[1]       + dz * up[2]
    z_cam = dx * forward[0] + dy * forward[1]  + dz * forward[2]

    if z_cam <= 1e-6:
        return 0.5, 0.5, False  # behind camera

    # focal length from horizontal FOV
    f = (screen_width / 2.0) / math.tan(math.radians(fov_deg / 2.0))

    u_px = f * x_cam / z_cam + screen_width  / 2.0
    v_px = screen_height / 2.0 - f * y_cam / z_cam  # image Y flipped

    u_norm = u_px / screen_width
    v_norm = v_px / screen_height

    in_frame = (0.0 <= u_norm <= 1.0) and (0.0 <= v_norm <= 1.0)
    return float(np.clip(u_norm, 0.0, 1.0)), float(np.clip(v_norm, 0.0, 1.0)), in_frame


# ── dataset helpers ───────────────────────────────────────────────────────────

def _load_dataset_index(dataset_dir: Path) -> dict[str, dict]:
    """
    Load all dataset entries indexed by (house_id, pair_id, direction, type).
    Returns flat dict keyed by (house_id, pair_id).
    """
    index: dict[tuple, dict] = {}
    house_files = sorted(dataset_dir.glob("train_house_*.json"))
    if not house_files:
        house_files = sorted((dataset_dir / "artifacts").glob("*/train_house_*.json"))
    for json_path in house_files:
        house_id = json_path.stem
        try:
            entries = json.loads(json_path.read_text(encoding="utf-8"))
            for e in (entries if isinstance(entries, list) else []):
                pair_id = e.get("pair_id", "")
                index[(house_id, pair_id)] = e
        except Exception as exc:
            print(f"  Warning: {json_path.name}: {exc}")
    return index


def _load_exploration_log(artifacts_dir: Path, house_id: str) -> dict[str, dict]:
    p = artifacts_dir / house_id / "exploration_log.json"
    if not p.exists():
        return {}
    log = json.loads(p.read_text(encoding="utf-8"))
    return {entry["image_path"]: entry for entry in log if "image_path" in entry}


# Anchor types whose top surface is also a valid lateral placement surface.
_GROUP_SPECIAL = {"Bed", "CounterTop", "Desk", "Desktop", "DiningTable", "Dresser", "SideTable"}

# Keywords used to match VLM surface strings to ProcTHOR anchor type names.
_ANCHOR_SURFACE_KEYWORDS: dict[str, list[str]] = {
    "Bed":          ["bed"],
    "CounterTop":   ["counter", "countertop"],
    "Desk":         ["desk"],
    "Desktop":      ["desktop"],
    "DiningTable":  ["dining table"],
    "Dresser":      ["dresser"],
    "SideTable":    ["side table"],
    "ArmChair":     ["armchair", "arm chair"],
    "Chair":        ["chair"],
    "CoffeeTable":  ["coffee table"],
    "Sofa":         ["sofa", "couch"],
    "Fridge":       ["fridge", "refrigerator"],
    "Sink":         ["sink"],
    "Toilet":       ["toilet"],
    "WashingMachine": ["washing machine"],
}


def _aabb_info(obj_meta: dict) -> tuple[float, float, float, float, float, float]:
    """Return (cx, cy, cz, hx, hz, top_y) from object_landmark metadata."""
    raw  = obj_meta.get("raw_metadata", {})
    aabb = raw.get("axisAlignedBoundingBox", {})
    ctr  = aabb.get("center") or obj_meta.get("position") or {}
    sz   = aabb.get("size", {})
    pts  = aabb.get("cornerPoints", [])
    cx   = float(ctr.get("x", 0))
    cy   = float(ctr.get("y", 0))
    cz   = float(ctr.get("z", 0))
    hx   = float(sz.get("x", 0)) / 2
    hz   = float(sz.get("z", 0)) / 2
    top_y = max(p[1] for p in pts) if pts else cy
    return cx, cy, cz, hx, hz, top_y


def _surface_matches_anchor(surface_lower: str, anchor_type: str) -> bool:
    """True when the VLM surface string refers to the anchor object itself."""
    keywords = _ANCHOR_SURFACE_KEYWORDS.get(anchor_type)
    if keywords is None:
        # Fallback: split CamelCase → words and check
        import re
        words = " ".join(re.findall("[A-Z][a-z]*", anchor_type)).lower()
        keywords = [words]
    return any(kw in surface_lower for kw in keywords)


def _y_from_surface(surface_lower: str, anchor_type: str, anchor_top_y: float) -> float:
    """
    Derive y_surface from VLM surface string.
      "floor"          → 0.0
      matches anchor   → anchor_top_y
      other            → anchor_top_y (best approximation; other objects not available)
    Final GT y = returned value + 0.2.
    """
    if "floor" in surface_lower:
        return 0.0
    if _surface_matches_anchor(surface_lower, anchor_type):
        return anchor_top_y
    # Surface names another object; use anchor top as approximation.
    return anchor_top_y


def _direction_vectors(entry: dict) -> dict[str, np.ndarray]:
    """
    Reconstruct the right/left/front unit vectors used during dataset generation.
    Follows the same convention as compute_direction_vectors() in dataset_generation_new3.py:
      forward = normalize(landmark_center − cam_pos) (horizontal only)
      right   = normalize(WORLD_UP × forward)
      left    = −right
      front   = −forward (from landmark toward camera)
    """
    cam   = entry.get("camera", {})
    cp    = cam.get("cam_pos", [0, 0, 0])
    cam_xz = np.array([float(cp[0]), 0.0, float(cp[2])], dtype=float)

    lm_pos = entry.get("object_landmark", {}).get("position", {})
    lm_xz  = np.array([float(lm_pos.get("x", 0)), 0.0, float(lm_pos.get("z", 0))], dtype=float)

    fwd = lm_xz - cam_xz
    fwd[1] = 0.0
    norm = float(np.linalg.norm(fwd))
    if norm > 1e-6:
        fwd /= norm

    WORLD_UP = np.array([0.0, 1.0, 0.0])
    right = np.cross(WORLD_UP, fwd)
    rn = float(np.linalg.norm(right))
    if rn > 1e-6:
        right /= rn

    return {"right": right, "left": -right, "front": -fwd}


def _bbox_boundary_xz(cx: float, cz: float, hx: float, hz: float,
                       d: np.ndarray, offset: float = 0.2) -> tuple[float, float]:
    """
    Ray from (cx, cz) in direction d; find first intersection with the AABB boundary,
    then step `offset` metres further. Returns (x, z).
    """
    dx, dz = float(d[0]), float(d[2])
    t_candidates = []
    if abs(dx) > 1e-6:
        t_candidates.append(hx / abs(dx))
    if abs(dz) > 1e-6:
        t_candidates.append(hz / abs(dz))
    t = min(t_candidates) if t_candidates else 0.3
    return cx + (t + offset) * dx, cz + (t + offset) * dz


def _compute_verified_gt_pos(entry: dict, direction: str) -> dict | None:
    """
    Compute corrected GT 3D position using VERIFY_DIRECTION surface verdict (stage3).

    Rules:
      UP (all anchors):
        x,z = anchor bbox centre;  y = anchor_top + 0.2

      FRONT (all anchors) and LEFT/RIGHT (anchor ∉ group_special):
        x,z = bbox-boundary intersection + 0.2 m offset
        y   = y_surface(VLM) + 0.2

      LEFT/RIGHT (anchor ∈ group_special):
       (a) VLM surface == anchor itself →
              x,z = bbox centre ± 0.2 m in left/right direction
              y   = anchor_top + 0.2
       (b) otherwise →
              x,z = bbox-boundary intersection + 0.2 m offset
              y   = y_surface(VLM) + 0.2
    """
    stage3 = entry.get("stage3", {})
    info   = stage3.get(direction)
    if not info or info.get("verdict") != "appropriate":
        return None

    surface      = (info.get("surface") or "").lower()
    anchor_type  = entry.get("landmark_type", "")
    lm_meta      = entry.get("object_landmark", {})
    cx, cy, cz, hx, hz, top_y = _aabb_info(lm_meta)

    Y_OFFSET = 0.2  # applied uniformly on top of y_surface

    # Case 1 — UP
    if direction == "up":
        return {"x": cx, "y": top_y + Y_OFFSET, "z": cz}

    dir_vecs = _direction_vectors(entry)
    d = dir_vecs.get(direction)
    if d is None:
        return None

    # Case 3 — LEFT / RIGHT for group_special anchors
    if direction in ("left", "right") and anchor_type in _GROUP_SPECIAL:
        if _surface_matches_anchor(surface, anchor_type):
            # (a) target is on the anchor's top surface, shifted left/right
            px = cx + float(d[0]) * Y_OFFSET
            pz = cz + float(d[2]) * Y_OFFSET
            return {"x": px, "y": top_y + Y_OFFSET, "z": pz}
        else:
            # (b) floor or other surface
            y_surf = _y_from_surface(surface, anchor_type, top_y)
            px, pz = _bbox_boundary_xz(cx, cz, hx, hz, d)
            return {"x": px, "y": y_surf + Y_OFFSET, "z": pz}

    # Case 2 — FRONT (all) or LEFT/RIGHT (anchor ∉ group_special)
    y_surf = _y_from_surface(surface, anchor_type, top_y)
    px, pz = _bbox_boundary_xz(cx, cz, hx, hz, d)
    return {"x": px, "y": y_surf + Y_OFFSET, "z": pz}


# ── evaluation ────────────────────────────────────────────────────────────────

def _evaluate_prediction(
    pred: dict,
    dataset_index: dict,
    artifacts_dir: Path,
    exploration_log_cache: dict,
    screen_width:  int = 512,
    screen_height: int = 512,
) -> dict:
    """
    Compute GT projection and error for one prediction row.
    Returns an evaluation dict with gt_position_2d, error metrics, etc.
    """
    result: dict[str, Any] = {
        "pair_id":    pred["pair_id"],
        "house_id":   pred["house_id"],
        "direction":  pred["direction"],
        "type":       pred["type"],
        "sentence":   pred.get("sentence", ""),
        "predicted_x": pred.get("predicted_x"),
        "predicted_y": pred.get("predicted_y"),
        "parse_error": pred.get("parse_error"),
    }

    # Skip if prediction missing
    if pred.get("predicted_x") is None or pred.get("predicted_y") is None:
        result["status"] = "no_prediction"
        return result

    house_id  = pred["house_id"]
    pair_id   = pred["pair_id"]
    direction = pred["direction"]
    sent_type = pred["type"]

    entry = dataset_index.get((house_id, pair_id))
    if entry is None:
        result["status"] = "entry_not_found"
        return result

    # GT 3D position — placements is top-level in dataset_generation_new3 output
    placements = entry.get("placements", {})
    placement  = placements.get(direction)
    if not placement:
        result["status"] = "placement_not_found"
        return result

    gt_pos_3d = placement.get("position")
    if not gt_pos_3d:
        result["status"] = "no_gt_position"
        return result

    corrected = _compute_verified_gt_pos(entry, direction)
    if corrected is not None:
        result["gt_surface_corrected"] = True
        gt_pos_3d = corrected
    else:
        result["gt_surface_corrected"] = False

    result["gt_position_3d"] = gt_pos_3d

    # Camera params depend on sentence type
    if sent_type == "c":
        # Type C uses the observer camera (stored at top-level "camera" key)
        cam = entry.get("camera", {})
        cam_pos     = cam.get("cam_pos", [0, 0, 0])
        cam_yaw     = cam.get("cam_yaw", 0.0)
        cam_horizon = cam.get("cam_horizon", 0.0)
        fov         = cam.get("fov", 100.0)
        sw          = cam.get("screen_width",  screen_width)
        sh          = cam.get("screen_height", screen_height)
        result["camera_image"] = entry.get("rgb_path")
    else:
        # Type A/B: use the VLM-selected exploration image
        predicted_image = pred.get("predicted_image")
        if not predicted_image:
            result["status"] = "no_selected_image"
            return result
        result["predicted_image"] = predicted_image

        if house_id not in exploration_log_cache:
            exploration_log_cache[house_id] = _load_exploration_log(artifacts_dir, house_id)
        elog = exploration_log_cache[house_id]

        log_entry = elog.get(predicted_image)
        if log_entry is None:
            result["status"] = "image_not_in_log"
            return result

        cam = log_entry.get("camera", {})
        cam_pos     = cam.get("cam_pos", [0, 0, 0])
        cam_yaw     = cam.get("cam_yaw", 0.0)
        cam_horizon = cam.get("cam_horizon", 0.0)
        fov         = cam.get("fov", 100.0)
        sw, sh      = screen_width, screen_height  # exploration images are always 512×512
        result["camera_image"] = predicted_image

        # also note whether selected image has both objects visible
        per_vis_path = artifacts_dir / house_id / "per_image_visible_objects.json"
        if per_vis_path.exists():
            per_vis = json.loads(per_vis_path.read_text())
            result["selected_image_types"] = per_vis.get(predicted_image, [])
            # candidate images from prediction
            result["candidate_images"] = pred.get("candidate_images", [])

    # Project GT 3D → 2D
    gx, gy, in_frame = project_3d_to_2d(
        gt_pos_3d, cam_pos, cam_yaw, cam_horizon,
        fov_deg=fov, screen_width=sw, screen_height=sh,
    )
    result["gt_position_2d"] = {"x": gx, "y": gy}
    result["gt_in_frame"] = in_frame

    # Normalize and clamp predictions to [0, 1], matching the ground truth.
    # This bounds out-of-frame predictions at the nearest image edge.
    px = float(np.clip(pred["predicted_x"] / sw, 0.0, 1.0))
    py = float(np.clip(pred["predicted_y"] / sh, 0.0, 1.0))
    result["predicted_x_norm"] = px
    result["predicted_y_norm"] = py

    ex = px - gx
    ey = py - gy
    el2 = math.sqrt(ex**2 + ey**2)

    result["error_x"]  = ex
    result["error_y"]  = ey
    result["error_l2"] = el2
    result["status"]   = "ok"

    # RoboPoint returns several points per item. Score two ways against the GT:
    #   error_l2_min  — the point CLOSEST to GT (best-case)
    #   error_l2_mean — the MEAN (centroid) of all points
    # Each point is normalized and clamped to [0, 1], symmetric with the GT above.
    pts = pred.get("predicted_points")
    if pts:
        norm = [(float(np.clip(p[0] / sw, 0.0, 1.0)), float(np.clip(p[1] / sh, 0.0, 1.0)))
                for p in pts]
        l2s = [math.hypot(nx - gx, ny - gy) for nx, ny in norm]
        cx = sum(n[0] for n in norm) / len(norm)
        cy = sum(n[1] for n in norm) / len(norm)
        result["n_points"]      = len(norm)
        result["error_l2_min"]  = min(l2s)
        result["error_l2_mean"] = math.hypot(cx - gx, cy - gy)
        # Primary error_l2 becomes the mean-point metric (both are kept above).
        result["error_l2"] = result["error_l2_mean"]

    return result


# ── stats & plots ─────────────────────────────────────────────────────────────

def _aggregate(evals: list[dict]) -> dict:
    ok = [e for e in evals if e.get("status") == "ok"]
    total = len(evals)
    n_ok  = len(ok)

    def _group_stats(group: list[dict]) -> dict:
        if not group:
            return {"n": 0, "mean_l2": None, "std_l2": None,
                    "mean_x": None, "mean_y": None, "pct_in_frame": None}
        l2s = [e["error_l2"] for e in group]
        exs = [e["error_x"]  for e in group]
        eys = [e["error_y"]  for e in group]
        inf = [e["gt_in_frame"] for e in group]
        return {
            "n":            len(group),
            "mean_l2":      float(np.mean(l2s)),
            "std_l2":       float(np.std(l2s)),
            "mean_abs_x":   float(np.mean(np.abs(exs))),
            "mean_abs_y":   float(np.mean(np.abs(eys))),
            "pct_in_frame": float(np.mean(inf)) * 100,
        }

    # L2 groups are built from answered (status == "ok") instances only, so the
    # mean L2 is unchanged by the response-rate addition below.
    per_type: dict[str, list] = {}
    per_dir:  dict[str, list] = {}
    for e in ok:
        per_type.setdefault(e["type"], []).append(e)
        per_dir.setdefault(e["direction"], []).append(e)

    # response rate = n_ok / n_total per group. Totals are counted over ALL
    # evals (including no_prediction / parse errors), which carry type & direction.
    total_per_type: dict[str, int] = {}
    total_per_dir:  dict[str, int] = {}
    for e in evals:
        if e.get("type") is not None:
            total_per_type[e["type"]] = total_per_type.get(e["type"], 0) + 1
        if e.get("direction") is not None:
            total_per_dir[e["direction"]] = total_per_dir.get(e["direction"], 0) + 1

    def _stats_with_rate(ok_group: list[dict], n_total: int) -> dict:
        s = _group_stats(ok_group)
        s["n_total"] = n_total
        s["response_rate"] = (len(ok_group) / n_total) if n_total else None
        return s

    overall = _stats_with_rate(ok, total)

    return {
        "total": total,
        "n_ok":  n_ok,
        "response_rate": (n_ok / total) if total else None,
        "n_parse_error": sum(1 for e in evals if e.get("parse_error")),
        "overall": overall,
        "per_type":      {k: _stats_with_rate(per_type.get(k, []), n)
                          for k, n in sorted(total_per_type.items())},
        "per_direction": {k: _stats_with_rate(per_dir.get(k, []), n)
                          for k, n in sorted(total_per_dir.items())},
    }


def _save_prediction_overlays(
    evals: list[dict],
    artifacts_dir: Path,
    output_dir: Path,
    dataset_index: dict | None = None,
) -> None:
    """Draw predicted (red ×) and GT (green ○) points on the selected image and save."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  Pillow not available — skipping prediction overlays")
        return

    overlay_dir = output_dir / "prediction_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for e in evals:
        if e.get("status") != "ok":
            continue
        img_rel = e.get("camera_image")
        if not img_rel:
            continue

        house_id = e.get("house_id", "")
        img_path = artifacts_dir / house_id / img_rel
        if not img_path.exists():
            img_path = artifacts_dir / img_rel
        if not img_path.exists():
            img_path = artifacts_dir.parent / img_rel
        if not img_path.exists():
            continue

        try:
            from PIL import ImageFont
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        w, h = img.size
        r = max(6, int(min(w, h) * 0.015))

        # GT point — green circle (skip if GT is behind the camera)
        if not e.get("gt_in_frame", True):
            continue
        gx = int(e["gt_position_2d"]["x"] * w)
        gy = int(e["gt_position_2d"]["y"] * h)

        # Predicted point
        px = int(e.get("predicted_x_norm", e["predicted_x"] / w) * w)
        py = int(e.get("predicted_y_norm", e["predicted_y"] / h) * h)

        # Add text label strip at the bottom
        label_h = 28
        canvas = Image.new("RGB", (w, h + label_h), (30, 30, 30))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)

        font = None
        for _font_name in ("DejaVuSans.ttf", "Helvetica.ttc", "Arial.ttf"):
            try:
                font = ImageFont.truetype(_font_name, 14)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        raw_px = int(e.get("predicted_x", px))
        raw_py = int(e.get("predicted_y", py))
        gt_text   = f"GT [{gx}, {gy}]"
        pred_text = f"Pred [{raw_px}, {raw_py}]"
        draw.text((6,  h + 6), gt_text,   fill="lime", font=font)
        draw.text((w // 2, h + 6), pred_text, fill="red",  font=font)

        # Draw markers on canvas
        draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline="lime", width=3)
        draw.line([gx - r, gy, gx + r, gy], fill="lime", width=2)
        draw.line([gx, gy - r, gx, gy + r], fill="lime", width=2)

        draw.line([px - r, py - r, px + r, py + r], fill="red", width=3)
        draw.line([px - r, py + r, px + r, py - r], fill="red", width=3)

        draw.line([gx, gy, px, py], fill="yellow", width=1)

        house_id  = e.get("house_id", "unknown")
        pair_id   = e.get("pair_id", "unknown").replace("|", "_")
        direction = e.get("direction", "")
        sent_type = e.get("type", "")

        anchor = ""
        if dataset_index is not None:
            entry = dataset_index.get((e.get("house_id", ""), e.get("pair_id", "")))
            if entry:
                lm = entry.get("object_landmark", {})
                anchor = (
                    lm.get("raw_metadata", {}).get("objectType")
                    or lm.get("assetId", "").split("_")[0]
                )
        anchor_part = f"_{anchor}" if anchor else ""
        fname = f"{house_id}__{pair_id}{anchor_part}_{direction}_{sent_type}.png"
        canvas.save(overlay_dir / fname)
        saved += 1

    print(f"  Saved {saved} prediction overlay images → {overlay_dir}")


def _make_plots(evals: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    ok = [e for e in evals if e.get("status") == "ok"]
    if not ok:
        print("  No valid predictions — skipping plots")
        return

    # ── 1. Bar chart: mean L2 error per sentence type ─────────────────────
    per_type: dict[str, list] = {}
    for e in ok:
        per_type.setdefault(e["type"], []).append(e["error_l2"])

    fig, ax = plt.subplots(figsize=(5, 4))
    types   = sorted(per_type.keys())
    means   = [float(np.mean(per_type[t])) for t in types]
    stds    = [float(np.std(per_type[t]))  for t in types]
    colors  = {"a": "#1d6f42", "b": "#1f4e79", "c": "#7b3f00"}
    bars = ax.bar(types, means, yerr=stds, capsize=5,
                  color=[colors.get(t, "#999") for t in types])
    ax.set_xlabel("Sentence type")
    ax.set_ylabel("L2 error (normalized, 0–1)")
    ax.set_title("Mean localization error by sentence type")
    ax.set_ylim(bottom=0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = output_dir / "error_by_type.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── 2. Bar chart: mean L2 error per direction ─────────────────────────
    per_dir: dict[str, list] = {}
    for e in ok:
        per_dir.setdefault(e["direction"], []).append(e["error_l2"])

    fig, ax = plt.subplots(figsize=(6, 4))
    dirs  = sorted(per_dir.keys())
    means = [float(np.mean(per_dir[d])) for d in dirs]
    stds  = [float(np.std(per_dir[d]))  for d in dirs]
    bars = ax.bar(dirs, means, yerr=stds, capsize=5, color="#4472C4")
    ax.set_xlabel("Direction")
    ax.set_ylabel("L2 error (normalized, 0–1)")
    ax.set_title("Mean localization error by direction")
    ax.set_ylim(bottom=0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = output_dir / "error_by_direction.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── 3. Scatter: predicted vs GT per type ──────────────────────────────
    fig, axes = plt.subplots(1, len(per_type), figsize=(4 * len(per_type), 4), squeeze=False)
    for ax, t in zip(axes[0], sorted(per_type.keys())):
        group = [e for e in ok if e["type"] == t]
        px = [e["predicted_x_norm"] for e in group]
        py = [e["predicted_y_norm"] for e in group]
        gx = [e["gt_position_2d"]["x"] for e in group]
        gy = [e["gt_position_2d"]["y"] for e in group]
        ax.scatter(gx, gy, alpha=0.6, s=30, label="GT",        color="#555")
        ax.scatter(px, py, alpha=0.6, s=30, label="Predicted",  color=colors.get(t, "#4472C4"), marker="x")
        for gxi, gyi, pxi, pyi in zip(gx, gy, px, py):
            ax.plot([gxi, pxi], [gyi, pyi], color="#ccc", linewidth=0.5, zorder=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("x (normalized)"); ax.set_ylabel("y (normalized)")
        ax.set_title(f"Type {t.upper()}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
    fig.suptitle("Predicted vs GT positions (image coords)")
    fig.tight_layout()
    p = output_dir / "scatter_predicted_vs_gt.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── 4. Combined bar: type × direction ─────────────────────────────────
    all_types = sorted({e["type"] for e in ok})
    all_dirs  = sorted({e["direction"] for e in ok})
    x = np.arange(len(all_dirs))
    width = 0.8 / max(len(all_types), 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, t in enumerate(all_types):
        means_d = []
        for d in all_dirs:
            vals = [e["error_l2"] for e in ok if e["type"] == t and e["direction"] == d]
            means_d.append(float(np.mean(vals)) if vals else 0.0)
        offset = (i - len(all_types) / 2 + 0.5) * width
        ax.bar(x + offset, means_d, width=width, label=f"Type {t.upper()}",
               color=colors.get(t, "#999"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_dirs)
    ax.set_xlabel("Direction")
    ax.set_ylabel("Mean L2 error (normalized)")
    ax.set_title("Localization error by direction × sentence type")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = output_dir / "error_by_direction_and_type.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")


# ── prediction loader ─────────────────────────────────────────────────────────

def _load_predictions(predictions_path: Path) -> tuple[list[dict], dict]:
    """
    Load predictions from either:
    - A directory containing per-house JSON files (each a plain list of dicts)
    - A single JSON file with {"metadata": ..., "predictions": [...]} or a plain list
    Returns (predictions_list, metadata_dict).
    """
    if predictions_path.is_dir():
        preds: list[dict] = []
        meta: dict = {}
        # accept both merged per-house lists AND raw worker shards, which are
        # dicts of the form {..., "predictions": [...]} (predictions_0.json, ...).
        # Restores the directory-loading behaviour these runs were evaluated with.
        # Skip our own eval outputs so a re-run never ingests them as predictions.
        _skip = {"meta.json", "eval_results.json"}
        for json_path in sorted(predictions_path.glob("*.json")):
            if json_path.name in _skip:
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"  Warning: {json_path.name}: {exc}")
                continue
            if isinstance(data, list):
                preds.extend(data)
            elif isinstance(data, dict):
                preds.extend(data.get("predictions", []))
                if not meta:
                    meta = data.get("metadata") or {
                        k: data[k] for k in ("screen_width", "screen_height", "vlm", "types")
                        if k in data
                    }
        return preds, meta
    else:
        pred_data = json.loads(predictions_path.read_text(encoding="utf-8"))
        if isinstance(pred_data, list):
            return pred_data, {}
        return pred_data.get("predictions", []), pred_data.get("metadata", {})


# ── main ─────────────────────────────────────────────────────────────────────

def evaluate(
    predictions_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    plot: bool = True,
) -> None:
    predictions, metadata = _load_predictions(predictions_path)

    screen_width  = metadata.get("screen_width",  512)
    screen_height = metadata.get("screen_height", 512)

    artifacts_dir = dataset_dir / "artifacts"
    dataset_index = _load_dataset_index(dataset_dir)
    exploration_log_cache: dict[str, dict] = {}

    print(f"Loaded {len(predictions)} predictions")
    print(f"Dataset index: {len(dataset_index)} entries")

    evals: list[dict] = []
    for pred in predictions:
        ev = _evaluate_prediction(
            pred, dataset_index, artifacts_dir, exploration_log_cache,
            screen_width=screen_width, screen_height=screen_height,
        )
        evals.append(ev)

    summary = _aggregate(evals)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full evaluation results
    eval_output = {
        "metadata": {
            **metadata,
            "dataset_dir_eval": str(dataset_dir),
        },
        "summary": summary,
        "evaluations": evals,
    }
    out_path = output_dir / "eval_results.json"
    out_path.write_text(json.dumps(eval_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved evaluation results → {out_path}")

    def _rr(s: dict) -> str:
        rr = s.get("response_rate")
        return "n/a" if rr is None else f"{rr * 100:.1f}%"

    # Print summary
    print("\n── Summary ──────────────────────────────────────")
    print(f"  Total predictions : {summary['total']}")
    print(f"  Successfully eval : {summary['n_ok']}")
    print(f"  Parse errors      : {summary['n_parse_error']}")
    print(f"  Response rate     : {_rr(summary)} ({summary['n_ok']}/{summary['total']})")
    ov = summary["overall"]
    if ov["n"]:
        print(f"  Overall mean L2   : {ov['mean_l2']:.4f} ± {ov['std_l2']:.4f} (clamped)")
        print(f"  GT in-frame       : {ov['pct_in_frame']:.1f}%")
    print("\n  Per sentence type:")
    for t, s in summary["per_type"].items():
        l2 = f"{s['mean_l2']:.4f} ± {s['std_l2']:.4f}" if s["n"] else "   n/a"
        print(f"    Type {t.upper()}: n={s['n']}/{s['n_total']}  mean_L2={l2}"
              f"  resp={_rr(s)}")
    print("\n  Per direction:")
    for d, s in summary["per_direction"].items():
        l2 = f"{s['mean_l2']:.4f} ± {s['std_l2']:.4f}" if s["n"] else "   n/a"
        print(f"    {d:8s}: n={s['n']}/{s['n_total']}  mean_L2={l2}  resp={_rr(s)}")

    if plot:
        print("\nGenerating plots...")
        _make_plots(evals, output_dir)
        print("\nGenerating prediction overlays...")
        _save_prediction_overlays(evals, artifacts_dir, output_dir, dataset_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM spatial reasoning predictions.")
    parser.add_argument("--predictions",  required=True, help="Directory of per-house prediction JSONs (from merge step), or a single predictions.json file.")
    parser.add_argument("--dataset-dir",  required=True, help="Path to the dataset directory.")
    parser.add_argument("--output-dir",   required=True, help="Directory to save eval_results.json and plots.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    evaluate(
        predictions_path=Path(args.predictions),
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
