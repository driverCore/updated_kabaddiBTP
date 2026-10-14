"""
classify_frame_conditions.py
============================
Frame-level difficulty classification for a fixed subset of diagnostic cases
(C01, C03, C04, C05, C06, C07, C10, C14, C19).

Every eligible frame is labelled on three INDEPENDENT factors, computed only from
ground-truth (GT) detections/identities and the original source JPGs. Heuristic
outputs (tracks_E*.json) are deliberately NOT read here, so the condition labels
cannot be influenced by heuristic success or failure.

  Factor A  Raider bounding-box overlap   max IoU(raider GT box, other GT boxes)
            -> HIGH_OVERLAP / LOW_OVERLAP / NOT_APPLICABLE / UNKNOWN
  Factor B  Raider image-plane speed      |d(centre)/dt| in px/s, dt = 1/video FPS
            -> HIGH_SPEED / LOW_SPEED / UNKNOWN
  Factor C  Raider relative camera depth  MiDaS (existing E8 implementation)
            -> FRONT / BEHIND / UNKNOWN / NOT_APPLICABLE / BLOCKED

Inputs (read-only):
  - Case windows/sequences: CASES_META in diagnostic_analysis/generate_gt_tool_cases.py
    (parsed with ast, not imported, so no tracker dependencies are loaded)
  - GT: dhruba_dataset_processed/<split>/<action>/bytetrack/bytetrack_<seq>_frames/tracks.json
  - Original frames: dhruba_dataset_processed/<split>/<action>/frames/<seq>_frames/frame_XXXX.jpg
  - FPS: dhruba_dataset_processed/<split>/<action>/videos/<seq>.mp4
  - MiDaS: tracking/E8_midas_depth/depth_features.py (load_midas, compute_depth_map)

Outputs: diagnostic_analysis/condition_classification/

Interpreter note: the MiDaS hub code needs `timm` (listed in tracking/requirements.txt).
mmaction_env does not have it; the system Python 3.13 (torch + timm + opencv) does.

Usage:
    python diagnostic_analysis/classify_frame_conditions.py
    python diagnostic_analysis/classify_frame_conditions.py --no-depth
    python diagnostic_analysis/classify_frame_conditions.py --config my_thresholds.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "dhruba_dataset_processed"
OUT_ROOT = SCRIPT_DIR / "condition_classification"
MIDAS_IMPL_DIR = PROJECT_ROOT / "tracking" / "E8_midas_depth"

# The nine requested cases. Resolved against folder names on disk (C1 -> C01).
REQUESTED_CASES = ["C1", "C3", "C4", "C5", "C6", "C7", "C10", "C14", "C19"]

# ---------------------------------------------------------------------------
# Threshold configuration (all overridable with --config JSON)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "overlap": {
        "high_overlap_iou_threshold": 0.25,
        "very_high_overlap_iou_threshold": 0.5,
        "comparison": "0.25 < IoU <= 0.5 -> HIGH_OVERLAP; IoU > 0.5 -> VERY_HIGH_OVERLAP",
        "source": "Reused from the project's earlier root-cause analysis "
                  "(results_analysis/evaluation/scripts/analyze_switch_reasons.py, "
                  "`max_box_overlap > 0.25` = occlusion trigger), which computed the same "
                  "quantity: max IoU between the GT raider box and any other GT box.",
        "exclude_other_raider_labelled_boxes": True,
    },
    "speed": {
        "high_speed_px_per_s_threshold": 1050.0,
        "comparison": "strictly greater than -> HIGH_SPEED",
        "source": "Reused from analyze_switch_reasons.py (`velocity > 35.0` px/frame, GT box-centre "
                  "displacement between consecutive frames), converted to px/s at a nominal 30 fps "
                  "(35 x 30 = 1050 px/s) so the threshold is a fixed physical-time rate. At the "
                  "actual case FPS (24.0-30.9) this equals 34.0-43.8 px/frame.",
        "max_gap_frames": 1,
        "max_gap_note": "Velocity is only computed between GT raider observations at most this many "
                        "frames apart. 1 = strictly consecutive frames; no interpolation across gaps.",
        "difference_scheme": "forward difference c[t+1]-c[t]; backward difference c[t]-c[t-1] only "
                             "when t+1 is unavailable (flagged SPEED_BACKWARD_DIFFERENCE)",
        "box_shape_change_flag_ratio": 0.5,
        "box_shape_change_note": "Flag (not relabel) when raider box area changes by more than this "
                                 "fraction between the two frames used: centre displacement then "
                                 "partly reflects box deformation (limb extension, crouch), not "
                                 "body translation. 0.5 matches the earlier project's 50% "
                                 "bbox-expansion rule.",
    },
    "depth": {
        "model": "MiDaS_small via existing tracking/E8_midas_depth/depth_features.py",
        "convention": "MiDaS outputs relative INVERSE depth (disparity-like): larger value = closer "
                      "to camera. Verified empirically per frame (see convention checks).",
        "statistic": "median of MiDaS values over each player's box pixels, EXCLUDING the pixels in "
                     "the raider/opponent box intersection (those pixels belong to whichever player "
                     "is in front and would bias both medians toward each other)",
        "normalisation": "difference divided by the frame's robust depth range (p95 - p5 of the "
                         "full depth map), because MiDaS scale is arbitrary per frame",
        "front_behind_margin": 0.05,
        "margin_source": "PROPOSED, not previously established in the repository. |normalised "
                         "difference| <= margin -> UNKNOWN. Sensitivity at 0.02 and 0.10 is "
                         "reported in classification_summary.md.",
        "sensitivity_margins": [0.02, 0.10],
        "min_valid_pixel_fraction": 0.20,
        "min_valid_pixels": 100,
        "valid_pixel_note": "If fewer than max(min_valid_pixels, fraction x box area) exclusive "
                            "pixels remain for either player, the label is UNKNOWN.",
        "opponent_selection": "the opponent with the greatest GT box IoU (> 0). If no box overlaps, "
                              "the nearest opponent by centre distance, if within "
                              "proximity_max_raider_heights x raider box height; otherwise "
                              "NOT_APPLICABLE.",
        "proximity_max_raider_heights": 1.0,
    },
    "duplicate_raider": {
        "rule": "When >1 GT box is labelled raider, pick the one with greatest IoU to the previous "
                "selected raider box (same rule as generate_gt_tool_cases.py).",
        "min_continuity_iou": 0.3,
        "min_continuity_margin": 0.2,
        "note": "If the best continuity IoU < min_continuity_iou or beats the runner-up by less "
                "than min_continuity_margin, the raider is ambiguous and the frame is INELIGIBLE.",
    },
}

CSV_FIELDS = [
    "case_id", "sequence", "split", "action", "frame_number", "frame_name", "frame_idx",
    "time_s", "original_jpg_path", "image_w", "image_h", "eligible", "ineligible_reason",
    "raider_gt_id", "raider_x1", "raider_y1", "raider_x2", "raider_y2", "raider_box_h",
    "n_gt_players", "n_raider_labelled_boxes", "n_comparison_players",
    # Factor A
    "max_overlap_iou", "max_overlap_player_id", "overlap_label",
    # Factor B
    "video_fps", "speed_px_per_s", "speed_px_per_frame", "speed_box_heights_per_s",
    "speed_scheme", "speed_dt_s", "accel_px_per_s2", "speed_label",
    # Factor C
    "depth_opponent_id", "depth_opponent_basis", "depth_opponent_iou",
    "depth_opponent_centre_dist_rh", "depth_raider_median", "depth_opponent_median",
    "depth_intersection_median", "depth_frame_range_p5_p95", "depth_diff_norm",
    "depth_raider_valid_px", "depth_opponent_valid_px", "depth_convention_rho",
    "depth_label",
    "flags",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def bbox_iou(b1: List[float], b2: List[float]) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def centre(b: List[float]) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def frame_number(name: str) -> int:
    m = re.search(r"(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse frame number from {name!r}")
    return int(m.group(1))


def rnd(v: Optional[float], nd: int = 4) -> Any:
    return "" if v is None else round(float(v), nd)


def load_cases_meta() -> Dict[str, Dict[str, Any]]:
    """Read CASES_META from the generator without importing it (avoids tracker deps)."""
    src = (SCRIPT_DIR / "generate_gt_tool_cases.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CASES_META" for t in node.targets
        ):
            return {m["case_id"]: m for m in ast.literal_eval(node.value)}
    raise RuntimeError("CASES_META not found in generate_gt_tool_cases.py")


def resolve_case_dirs(requested: List[str]) -> Dict[str, str]:
    """Map requested IDs (e.g. 'C1') to case folder names on disk (e.g. 'C01')."""
    on_disk = {int(p.name[1:]): p.name for p in SCRIPT_DIR.iterdir()
               if p.is_dir() and re.fullmatch(r"C\d+", p.name)}
    out = {}
    for r in requested:
        n = int(r[1:])
        if n not in on_disk:
            raise FileNotFoundError(f"No case folder for {r} under {SCRIPT_DIR}")
        out[r] = on_disk[n]
    return out


# ---------------------------------------------------------------------------
# GT loading and raider selection
# ---------------------------------------------------------------------------
def case_paths(meta: Dict[str, Any]) -> Dict[str, Path]:
    base = DATA_ROOT / meta["split"] / meta["action"]
    seq = meta["seq"]
    return {
        "gt": base / "bytetrack" / f"bytetrack_{seq}_frames" / "tracks.json",
        "frames": base / "frames" / f"{seq}_frames",
        "video": base / "videos" / f"{seq}.mp4",
    }


def is_raider(t: Dict[str, Any]) -> bool:
    # class_name is authoritative (same rule as the generator)
    cname = t.get("class_name")
    return cname == "raider" if cname else t.get("class_id") == 0


def select_raiders(gt_records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Walk the full source sequence in frame order and pick one raider box per frame.
    Returns {frame_number: {"box", "gt_id", "n_raider_boxes", "ambiguous", "cont_ious"}}.
    """
    dcfg = cfg["duplicate_raider"]
    out: Dict[int, Dict[str, Any]] = {}
    prev_box: Optional[List[float]] = None
    for rec in sorted(gt_records, key=lambda r: frame_number(r["frame"])):
        fn = frame_number(rec["frame"])
        raiders = [t for t in rec["tracks"] if is_raider(t) and t.get("bbox_xyxy") and len(t["bbox_xyxy"]) == 4]
        if not raiders:
            prev_box = None  # do not carry identity across a raider gap
            continue
        info: Dict[str, Any] = {"n_raider_boxes": len(raiders), "ambiguous": False, "cont_ious": []}
        if len(raiders) == 1:
            chosen = raiders[0]
        else:
            if prev_box is None:
                chosen, info["ambiguous"] = raiders[0], True
            else:
                ious = sorted(((bbox_iou(t["bbox_xyxy"], prev_box), i) for i, t in enumerate(raiders)), reverse=True)
                info["cont_ious"] = [round(v, 3) for v, _ in ious]
                chosen = raiders[ious[0][1]]
                if ious[0][0] < dcfg["min_continuity_iou"] or ious[0][0] - ious[1][0] < dcfg["min_continuity_margin"]:
                    info["ambiguous"] = True
        info["box"] = [float(v) for v in chosen["bbox_xyxy"]]
        info["gt_id"] = int(chosen.get("track_id", chosen.get("original_id")))
        info["track_ref"] = chosen
        out[fn] = info
        prev_box = info["box"]
    return out


# ---------------------------------------------------------------------------
# Video / frame validation
# ---------------------------------------------------------------------------
def video_info(video: Path) -> Tuple[float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return float("nan"), -1
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def verify_frame_correspondence(video: Path, jpgs: Dict[int, Path], frame_idx: Dict[int, int]) -> Dict[str, Any]:
    """
    Decode the video and check that each window JPG is closest (mean abs diff) to the
    decoded frame at its own frame_idx, rather than to its neighbours.
    """
    wanted = sorted(frame_idx.values())
    if not wanted:
        return {"checked": 0}
    lo, hi = max(0, wanted[0] - 1), wanted[-1] + 1
    cap = cv2.VideoCapture(str(video))
    decoded: Dict[int, np.ndarray] = {}
    i = 0
    while i <= hi:
        ok, fr = cap.read()
        if not ok:
            break
        if i >= lo:
            decoded[i] = cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
        i += 1
    cap.release()
    res = {"checked": 0, "matched_own_index": 0, "ties_or_static": 0, "mismatch_frames": [], "mean_abs_diff_own": []}
    for fn, idx in frame_idx.items():
        img = cv2.imread(str(jpgs[fn]))
        if img is None or idx not in decoded:
            res["mismatch_frames"].append(fn)
            continue
        small = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
        own = float(np.abs(small - decoded[idx]).mean())
        neigh = [float(np.abs(small - decoded[j]).mean()) for j in (idx - 1, idx + 1) if j in decoded]
        res["checked"] += 1
        res["mean_abs_diff_own"].append(own)
        if all(own < n for n in neigh):
            res["matched_own_index"] += 1
        elif all(own <= n for n in neigh):
            res["ties_or_static"] += 1
        else:
            res["mismatch_frames"].append(fn)
    m = res.pop("mean_abs_diff_own")
    res["max_mean_abs_diff_own"] = round(max(m), 3) if m else None
    return res


# ---------------------------------------------------------------------------
# Factor C helpers (MiDaS)
# ---------------------------------------------------------------------------
class DepthEstimator:
    """Thin wrapper around the existing E8 MiDaS helpers; no new model is introduced."""

    def __init__(self):
        sys.path.insert(0, str(MIDAS_IMPL_DIR))
        from depth_features import compute_depth_map, load_midas  # existing implementation
        self._compute = compute_depth_map
        self.model, self.transform, self.device = load_midas()

    def depth_map(self, img_bgr: np.ndarray, frame: Optional[int] = None) -> np.ndarray:
        return self._compute(img_bgr, self.model, self.transform, self.device)


def box_mask(shape: Tuple[int, int], b: List[float]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    x1, y1 = max(0, int(round(b[0]))), max(0, int(round(b[1])))
    x2, y2 = min(w, int(round(b[2]))), min(h, int(round(b[3])))
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def depth_compare(dm: np.ndarray, raider: List[float], opp: List[float], cfg: Dict[str, Any]) -> Dict[str, Any]:
    shape = dm.shape[:2]
    mr, mo = box_mask(shape, raider), box_mask(shape, opp)
    inter = mr & mo
    r_ex, o_ex = mr & ~inter, mo & ~inter
    p5, p95 = np.percentile(dm, [5, 95])
    rng = float(p95 - p5)
    out = {
        "r_valid": int(r_ex.sum()), "o_valid": int(o_ex.sum()),
        "r_med": None, "o_med": None, "i_med": float(np.median(dm[inter])) if inter.any() else None,
        "range": rng, "diff_norm": None, "reason": None,
    }
    need_r = max(cfg["min_valid_pixels"], cfg["min_valid_pixel_fraction"] * int(mr.sum()))
    need_o = max(cfg["min_valid_pixels"], cfg["min_valid_pixel_fraction"] * int(mo.sum()))
    if out["r_valid"] < need_r or out["o_valid"] < need_o:
        out["reason"] = "DEPTH_INSUFFICIENT_EXCLUSIVE_PIXELS"
        return out
    if rng <= 1e-6:
        out["reason"] = "DEPTH_FLAT_FRAME"
        return out
    out["r_med"] = float(np.median(dm[r_ex]))
    out["o_med"] = float(np.median(dm[o_ex]))
    # inverse depth: larger = closer -> positive diff means raider closer
    out["diff_norm"] = (out["r_med"] - out["o_med"]) / rng
    return out


def depth_label(diff_norm: Optional[float], margin: float) -> str:
    if diff_norm is None or not math.isfinite(diff_norm):
        return "UNKNOWN"
    if diff_norm > margin:
        return "FRONT"
    if diff_norm < -margin:
        return "BEHIND"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------
def process_case(req_id: str, case_id: str, meta: Dict[str, Any], cfg: Dict[str, Any],
                 depth: Optional[DepthEstimator], depth_block_reason: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    paths = case_paths(meta)
    pre: Dict[str, Any] = {"case_id": case_id, "requested_as": req_id, "sequence": meta["seq"],
                           "window": [meta["start_frame"], meta["end_frame"]], "issues": []}
    for k, p in paths.items():
        if not p.exists():
            pre["issues"].append(f"missing {k}: {p}")
    if pre["issues"]:
        return [], pre

    gt = json.loads(paths["gt"].read_text(encoding="utf-8"))
    # Schema check
    for rec in gt:
        for key in ("frame", "frame_idx", "image_w", "image_h", "tracks"):
            if key not in rec:
                pre["issues"].append(f"GT record missing key {key!r}")
                break
        for t in rec.get("tracks", []):
            if not (t.get("bbox_xyxy") and len(t["bbox_xyxy"]) == 4):
                pre["issues"].append(f"{rec.get('frame')}: track without valid bbox_xyxy")
    nums = [frame_number(r["frame"]) for r in gt]
    pre["gt_records"] = len(gt)
    pre["duplicate_frame_numbers"] = sorted(n for n, c in Counter(nums).items() if c > 1)
    pre["frame_idx_consistent"] = all(int(r["frame_idx"]) == frame_number(r["frame"]) - 1 for r in gt)
    by_num = {frame_number(r["frame"]): r for r in gt}
    window = list(range(meta["start_frame"], meta["end_frame"] + 1))
    pre["window_frames_missing_in_gt"] = [n for n in window if n not in by_num]

    jpgs = {n: paths["frames"] / f"{by_num[n]['frame']}.jpg" for n in window if n in by_num}
    pre["window_jpgs_missing"] = [n for n, p in jpgs.items() if not p.is_file()]

    fps, n_video = video_info(paths["video"])
    n_jpg = len(list(paths["frames"].glob("*.jpg")))
    pre["video_fps"] = round(fps, 4)
    pre["video_frame_count"] = n_video
    pre["source_jpg_count"] = n_jpg
    pre["full_rate_extraction"] = (n_video == n_jpg)
    fps_ok = math.isfinite(fps) and fps > 0 and n_video == n_jpg
    if not fps_ok:
        pre["issues"].append("FPS unreliable or JPG count != video frame count; speed set UNKNOWN")
    pre["frame_correspondence"] = verify_frame_correspondence(
        paths["video"], jpgs, {n: int(by_num[n]["frame_idx"]) for n in jpgs})

    raiders = select_raiders(gt, cfg)
    raider_ids = sorted({raiders[n]["gt_id"] for n in window if n in raiders})
    pre["raider_gt_ids_in_window"] = raider_ids
    pre["raider_missing_frames"] = [n for n in window if n in by_num and n not in raiders]
    pre["duplicate_raider_frames"] = [n for n in window if n in raiders and raiders[n]["n_raider_boxes"] > 1]

    ocfg, scfg, dcfg = cfg["overlap"], cfg["speed"], cfg["depth"]
    rows: List[Dict[str, Any]] = []
    for n in window:
        rec = by_num.get(n)
        row: Dict[str, Any] = {k: "" for k in CSV_FIELDS}
        flags: List[str] = []
        row.update(case_id=case_id, sequence=meta["seq"], split=meta["split"], action=meta["action"],
                   frame_number=n, video_fps=rnd(fps))
        if rec is None:
            row.update(eligible=0, ineligible_reason="NO_GT_RECORD", overlap_label="UNKNOWN",
                       speed_label="UNKNOWN", depth_label="UNKNOWN")
            rows.append(row)
            continue
        jpg = jpgs[n]
        row.update(frame_name=rec["frame"], frame_idx=int(rec["frame_idx"]),
                   time_s=rnd(int(rec["frame_idx"]) / fps if fps_ok else None, 5),
                   original_jpg_path=str(jpg.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                   image_w=rec["image_w"], image_h=rec["image_h"], n_gt_players=len(rec["tracks"]))
        r = raiders.get(n)
        reason = None
        if not jpg.is_file():
            reason = "ORIGINAL_JPG_MISSING"
        elif r is None:
            reason = "NO_RAIDER_GT_BOX"
        elif r["ambiguous"]:
            reason = "AMBIGUOUS_DUPLICATE_RAIDER_GT"
        if r is not None:
            row["n_raider_labelled_boxes"] = r["n_raider_boxes"]
            if r["n_raider_boxes"] > 1:
                flags.append(f"DUPLICATE_RAIDER_GT_RESOLVED_BY_CONTINUITY(ious={r['cont_ious']})")
        else:
            row["n_raider_labelled_boxes"] = 0
        if reason:
            row.update(eligible=0, ineligible_reason=reason, overlap_label="UNKNOWN",
                       speed_label="UNKNOWN", depth_label="UNKNOWN", flags=";".join(flags))
            rows.append(row)
            continue

        rb = r["box"]
        rh = rb[3] - rb[1]
        row.update(eligible=1, raider_gt_id=r["gt_id"], raider_x1=rnd(rb[0], 2), raider_y1=rnd(rb[1], 2),
                   raider_x2=rnd(rb[2], 2), raider_y2=rnd(rb[3], 2), raider_box_h=rnd(rh, 2))
        if r["track_ref"].get("status") != "edited":
            flags.append(f"RAIDER_GT_STATUS_{r['track_ref'].get('status')}")

        # ---------------- Factor A: overlap ----------------
        others = [t for t in rec["tracks"] if t is not r["track_ref"]
                  and not (ocfg["exclude_other_raider_labelled_boxes"] and is_raider(t))]
        row["n_comparison_players"] = len(others)
        if not others:
            row["overlap_label"] = "NOT_APPLICABLE"
        else:
            ious = [(bbox_iou(rb, [float(v) for v in t["bbox_xyxy"]]), t) for t in others]
            best_iou, best_t = max(ious, key=lambda x: x[0])
            row["max_overlap_iou"] = rnd(best_iou)
            row["max_overlap_player_id"] = int(best_t.get("track_id", best_t.get("original_id")))
            row["overlap_label"] = ("VERY_HIGH_OVERLAP" if best_iou > ocfg["very_high_overlap_iou_threshold"]
                                    else "HIGH_OVERLAP" if best_iou > ocfg["high_overlap_iou_threshold"]
                                    else "LOW_OVERLAP")

        # ---------------- Factor B: speed ----------------
        if not fps_ok:
            row["speed_label"] = "UNKNOWN"
            flags.append("SPEED_FPS_UNRELIABLE")
        else:
            nxt = next((k for k in range(n + 1, n + 1 + scfg["max_gap_frames"])
                        if k in raiders and not raiders[k]["ambiguous"]), None)
            prv = next((k for k in range(n - 1, n - 1 - scfg["max_gap_frames"], -1)
                        if k in raiders and not raiders[k]["ambiguous"]), None)
            if nxt is not None:
                a, b, scheme = n, nxt, "forward"
            elif prv is not None:
                a, b, scheme = prv, n, "backward"
                flags.append("SPEED_BACKWARD_DIFFERENCE")
            else:
                a = b = None
                scheme = ""
            if a is None:
                row["speed_label"] = "UNKNOWN"
                flags.append("SPEED_NO_ADJACENT_RAIDER_OBSERVATION")
            else:
                dt = (b - a) / fps
                (ax, ay), (bx, by) = centre(raiders[a]["box"]), centre(raiders[b]["box"])
                vx, vy = (bx - ax) / dt, (by - ay) / dt
                sp = math.hypot(vx, vy)
                row.update(speed_px_per_s=rnd(sp, 3), speed_px_per_frame=rnd(sp / fps, 3),
                           speed_box_heights_per_s=rnd(sp / rh if rh > 0 else None, 4),
                           speed_scheme=scheme, speed_dt_s=rnd(dt, 6))
                row["speed_label"] = "HIGH_SPEED" if sp > scfg["high_speed_px_per_s_threshold"] else "LOW_SPEED"
                if b - a > 1:
                    flags.append(f"SPEED_ACROSS_GAP_{b - a}_FRAMES")
                aa, ab = area(raiders[a]["box"]), area(raiders[b]["box"])
                if aa > 0 and abs(ab - aa) / aa > scfg["box_shape_change_flag_ratio"]:
                    flags.append("SPEED_BOX_SHAPE_CHANGE")
                # Supplementary: acceleration magnitude (central second difference), consecutive only
                if (n - 1) in raiders and (n + 1) in raiders and not raiders[n - 1]["ambiguous"] \
                        and not raiders[n + 1]["ambiguous"]:
                    (px, py), (cx, cy), (nx, ny) = centre(raiders[n - 1]["box"]), centre(rb), centre(raiders[n + 1]["box"])
                    row["accel_px_per_s2"] = rnd(math.hypot(nx - 2 * cx + px, ny - 2 * cy + py) * fps * fps, 2)

        # ---------------- Factor C: depth ----------------
        if depth is None:
            row["depth_label"] = "BLOCKED"
            flags.append(f"DEPTH_BLOCKED({depth_block_reason})")
        elif not others:
            row["depth_label"] = "NOT_APPLICABLE"
        else:
            cand = [(bbox_iou(rb, [float(v) for v in t["bbox_xyxy"]]), t) for t in others]
            ov = [c for c in cand if c[0] > 0]
            if ov:
                opp_iou, opp = max(ov, key=lambda x: x[0])
                basis = "max_iou"
            else:
                rc = centre(rb)
                dist = [(math.dist(rc, centre([float(v) for v in t["bbox_xyxy"]])) / rh, t) for _, t in cand]
                dn, opp = min(dist, key=lambda x: x[0])
                opp_iou, basis = 0.0, "nearest"
                row["depth_opponent_centre_dist_rh"] = rnd(dn)
                if dn > dcfg["proximity_max_raider_heights"]:
                    opp = None
            if opp is None:
                row["depth_label"] = "NOT_APPLICABLE"
                flags.append("DEPTH_NO_OPPONENT_WITHIN_PROXIMITY")
            else:
                ob = [float(v) for v in opp["bbox_xyxy"]]
                img = cv2.imread(str(jpg))
                if img is None or img.shape[1] != rec["image_w"] or img.shape[0] != rec["image_h"]:
                    row["depth_label"] = "UNKNOWN"
                    flags.append("DEPTH_IMAGE_UNREADABLE_OR_SIZE_MISMATCH")
                else:
                    dm = depth.depth_map(img, frame=n)
                    if dm.shape[:2] != img.shape[:2]:
                        row["depth_label"] = "UNKNOWN"
                        flags.append("DEPTH_MAP_SIZE_MISMATCH")
                    else:
                        res = depth_compare(dm, rb, ob, dcfg)
                        # convention check: lower feet (larger y2) should be closer (larger value)
                        feet, meds = [], []
                        for t in rec["tracks"]:
                            tb = [float(v) for v in t["bbox_xyxy"]]
                            mk = box_mask(dm.shape[:2], tb)
                            if mk.any():
                                feet.append(tb[3])
                                meds.append(float(np.median(dm[mk])))
                        rho = spearman(feet, meds)
                        row.update(
                            depth_opponent_id=int(opp.get("track_id", opp.get("original_id"))),
                            depth_opponent_basis=basis, depth_opponent_iou=rnd(opp_iou),
                            depth_raider_median=rnd(res["r_med"]), depth_opponent_median=rnd(res["o_med"]),
                            depth_intersection_median=rnd(res["i_med"]),
                            depth_frame_range_p5_p95=rnd(res["range"]), depth_diff_norm=rnd(res["diff_norm"], 5),
                            depth_raider_valid_px=res["r_valid"], depth_opponent_valid_px=res["o_valid"],
                            depth_convention_rho=rnd(rho),
                            depth_label=depth_label(res["diff_norm"], dcfg["front_behind_margin"]),
                        )
                        if res["reason"]:
                            flags.append(res["reason"])
                        elif row["depth_label"] == "UNKNOWN":
                            flags.append("DEPTH_DIFF_WITHIN_MARGIN")
        row["flags"] = ";".join(flags)
        rows.append(row)
    return rows, pre


# ---------------------------------------------------------------------------
# Post-run validation and summary
# ---------------------------------------------------------------------------
def post_validate(rows: List[Dict[str, Any]], case_ids: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    v: Dict[str, Any] = {}
    elig = [r for r in rows if r["eligible"] == 1]
    v["only_requested_cases"] = sorted({r["case_id"] for r in rows}) == sorted(case_ids)
    v["all_rows_have_existing_original_jpg"] = all(
        (PROJECT_ROOT / r["original_jpg_path"]).is_file() for r in rows if r["original_jpg_path"])
    v["original_jpgs_outside_diagnostic_analysis"] = all(
        not r["original_jpg_path"].startswith("diagnostic_analysis") for r in rows if r["original_jpg_path"])
    ious = [float(r["max_overlap_iou"]) for r in elig if r["max_overlap_iou"] != ""]
    v["iou_in_0_1"] = all(0.0 <= x <= 1.0 for x in ious)
    sp = [float(r["speed_px_per_s"]) for r in elig if r["speed_px_per_s"] != ""]
    v["speed_finite_nonnegative"] = all(math.isfinite(x) and x >= 0 for x in sp)
    m = cfg["depth"]["front_behind_margin"]
    ok = True
    for r in elig:
        d = r["depth_diff_norm"]
        lab = r["depth_label"]
        tol = 5e-6  # depth_diff_norm is stored rounded to 5 decimals; labels use the unrounded value
        if lab == "FRONT" and not (d != "" and float(d) > m - tol):
            ok = False
        if lab == "BEHIND" and not (d != "" and float(d) < -m + tol):
            ok = False
        if d == "" and lab in ("FRONT", "BEHIND"):
            ok = False
    v["depth_labels_follow_inverse_depth_convention"] = ok
    v["ineligible_rows_have_no_factor_labels"] = all(
        r["overlap_label"] == r["speed_label"] == r["depth_label"] == "UNKNOWN" for r in rows if r["eligible"] == 0)
    rhos = [float(r["depth_convention_rho"]) for r in elig if r["depth_convention_rho"] != ""]
    v["convention_rho_frames"] = len(rhos)
    v["convention_rho_positive_frames"] = sum(1 for x in rhos if x > 0)
    v["convention_rho_median"] = round(float(np.median(rhos)), 3) if rhos else None
    return v


def counts(rows: List[Dict[str, Any]], key: str) -> Counter:
    return Counter(r[key] for r in rows if r["eligible"] == 1)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_summary(all_rows: List[Dict[str, Any]], prechecks: List[Dict[str, Any]], post: Dict[str, Any],
                  cfg: Dict[str, Any], depth_status: str) -> str:
    L: List[str] = []
    L.append("# Frame-Condition Classification — Selected Diagnostic Cases\n")
    L.append("Condition-wise diagnostic classification on 9 selected cases (C01, C03, C04, C05, C06, C07, "
             "C10, C14, C19). Not a full-dataset benchmark. Labels are computed from GT boxes/identities, "
             "original source JPGs and MiDaS depth only; heuristic outputs are not read by "
             "`classify_frame_conditions.py`.\n")
    L.append(f"**Depth factor status:** {depth_status}\n")

    L.append("## Frames per case\n")
    prox = cfg["depth"]["proximity_max_raider_heights"]

    def na_reason(r: Dict[str, Any]) -> str:
        if r["n_comparison_players"] in (0, ""):
            return f"f{r['frame_number']}: no other GT player in frame"
        return (f"f{r['frame_number']}: no overlapping opponent; nearest is "
                f"{r['depth_opponent_centre_dist_rh']} raider-heights away (> {prox})")

    L.append("| Case | Sequence | Window | Frames | Eligible | HIGH_OVERLAP | LOW_OVERLAP | "
             "HIGH_SPEED | LOW_SPEED | FRONT | BEHIND | Depth UNKNOWN | Depth N/A (reason) |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    tot = Counter()
    for pc in prechecks:
        cr = [r for r in all_rows if r["case_id"] == pc["case_id"]]
        el = [r for r in cr if r["eligible"] == 1]
        o, s, d = counts(cr, "overlap_label"), counts(cr, "speed_label"), counts(cr, "depth_label")
        vals = dict(frames=len(cr), elig=len(el), ho=o["HIGH_OVERLAP"], lo=o["LOW_OVERLAP"], hs=s["HIGH_SPEED"],
                    ls=s["LOW_SPEED"], fr=d["FRONT"], be=d["BEHIND"], du=d["UNKNOWN"] + d["BLOCKED"],
                    dn=d["NOT_APPLICABLE"])
        tot.update(vals)
        na = [na_reason(r) for r in el if r["depth_label"] == "NOT_APPLICABLE"]
        na_s = f"{vals['dn']} — " + "; ".join(na) if na else "0"
        L.append(f"| {pc['case_id']} | `{pc['sequence']}` | {pc['window'][0]}–{pc['window'][1]} | {vals['frames']} | "
                 f"{vals['elig']} | {vals['ho']} | {vals['lo']} | {vals['hs']} | {vals['ls']} | "
                 f"{vals['fr']} | {vals['be']} | {vals['du']} | {na_s} |")
    L.append(f"| **Total** | | | **{tot['frames']}** | **{tot['elig']}** | "
             f"**{tot['ho']}** | **{tot['lo']}** | **{tot['hs']}** | **{tot['ls']}** | **{tot['fr']}** | "
             f"**{tot['be']}** | **{tot['du']}** | **{tot['dn']}** |\n")
    L.append(f"Depth N/A means no eligible comparison opponent: no GT box overlaps the raider (IoU = 0) and the "
             f"nearest player's box centre is more than {prox} raider box-height(s) from the raider's centre, so "
             f"there is no relevant opponent whose depth order matters. Frames minus Eligible = frames excluded "
             f"for missing GT raider (C05 frames 81–87).\n")

    L.append("## Missing-data and reliability flags (eligible frames)\n")
    fc = Counter()
    for r in all_rows:
        for f in filter(None, r["flags"].split(";")):
            fc[re.sub(r"\(.*\)", "", f)] += 1
    L.append("| Flag | Frames |\n|---|---:|")
    for k, v in sorted(fc.items()):
        L.append(f"| `{k}` | {v} |")
    L.append("")

    # Depth margin sensitivity
    L.append("## Depth margin sensitivity (same depth values, different margins)\n")
    L.append("| Margin | FRONT | BEHIND | UNKNOWN (within margin or insufficient pixels) |\n|---:|---:|---:|---:|")
    el = [r for r in all_rows if r["eligible"] == 1 and r["depth_label"] not in ("NOT_APPLICABLE", "BLOCKED")]
    for m in sorted({cfg["depth"]["front_behind_margin"], *cfg["depth"]["sensitivity_margins"]}):
        c = Counter(depth_label(float(r["depth_diff_norm"]) if r["depth_diff_norm"] != "" else None, m) for r in el)
        star = " (used)" if m == cfg["depth"]["front_behind_margin"] else ""
        L.append(f"| {m}{star} | {c['FRONT']} | {c['BEHIND']} | {c['UNKNOWN']} |")
    L.append("")

    L.append("## Thresholds and rationale\n")
    L.append(f"- **Overlap:** HIGH_OVERLAP if max IoU > **{cfg['overlap']['high_overlap_iou_threshold']}**. "
             f"{cfg['overlap']['source']}")
    L.append(f"- **Speed:** HIGH_SPEED if speed > **{cfg['speed']['high_speed_px_per_s_threshold']} px/s**. "
             f"{cfg['speed']['source']} Velocity uses Δt = 1/FPS of the source video (FPS read with OpenCV; "
             f"JPG count equals video frame count for every case, so frames are full-rate). "
             f"{cfg['speed']['difference_scheme']}.")
    L.append(f"- **Depth:** FRONT/BEHIND if |normalised median-depth difference| > **{cfg['depth']['front_behind_margin']}**. "
             f"{cfg['depth']['margin_source']} Statistic: {cfg['depth']['statistic']}; {cfg['depth']['normalisation']}.")
    L.append("- Full configuration: `threshold_configuration.json`.\n")

    L.append("## Validation\n")
    L.append("### Pre-run (per case)\n")
    L.append("| Case | GT records | Window frames missing in GT | JPGs missing | Dup. frame nos. | frame_idx = n−1 | "
             "Video FPS | Video frames = JPGs | JPG↔video frame match | Raider GT IDs | Raider-missing frames | "
             "Duplicate-raider frames |")
    L.append("|---|---:|---|---|---|---|---:|---|---|---|---|---|")
    for pc in prechecks:
        fcv = pc.get("frame_correspondence", {})
        L.append(f"| {pc['case_id']} | {pc.get('gt_records')} | {pc.get('window_frames_missing_in_gt') or '—'} | "
                 f"{pc.get('window_jpgs_missing') or '—'} | {pc.get('duplicate_frame_numbers') or '—'} | "
                 f"{pc.get('frame_idx_consistent')} | {pc.get('video_fps')} | {pc.get('full_rate_extraction')} "
                 f"({pc.get('video_frame_count')}/{pc.get('source_jpg_count')}) | "
                 f"{fcv.get('matched_own_index')}/{fcv.get('checked')} closest to own index"
                 f"{', ties ' + str(fcv.get('ties_or_static')) if fcv.get('ties_or_static') else ''}"
                 f"{', MISMATCH ' + str(fcv.get('mismatch_frames')) if fcv.get('mismatch_frames') else ''} | "
                 f"{pc.get('raider_gt_ids_in_window')} | {pc.get('raider_missing_frames') or '—'} | "
                 f"{pc.get('duplicate_raider_frames') or '—'} |")
    L.append("\n### Post-run\n")
    for k, v in post.items():
        L.append(f"- `{k}`: {v}")
    L.append("")

    L.append("## Limitations\n")
    L.append("- **Overlap** is 2D bounding-box IoU, not physical occlusion: boxes can overlap with no visual "
             "occlusion (players side by side in depth) and limbs can occlude with low box IoU. Defender boxes "
             "in the source GT are mostly `unreviewed`; only raider boxes were edited.")
    L.append("- **Speed** is image-plane speed of the GT box centre in px/s. It mixes player motion with camera "
             "pan/zoom (broadcast footage), depends on resolution (cases range 1122–2400 px wide) and on "
             "distance to camera (perspective). Box deformation (crouch, limb extension) moves the centre "
             "without body translation (flagged `SPEED_BOX_SHAPE_CHANGE`). `speed_box_heights_per_s` is "
             "provided as a scale-normalised alternative but is not used for the label. Acceleration "
             "(`accel_px_per_s2`, central second difference — the velocity-difference term in the notes) is "
             "recorded but not used for classification.")
    L.append("- **Depth** is monocular relative inverse depth from MiDaS_small; scale is arbitrary per frame. "
             "Box medians include background/mat pixels and the opponent's limbs outside the intersection. "
             "In near-total box containment the exclusive region is too small and the label is UNKNOWN. "
             "The margin is a proposed value (sensitivity reported above). For non-overlapping frames the "
             "comparison is against the nearest opponent within one raider-height, so FRONT/BEHIND there "
             "describes depth order, not occlusion.")
    L.append("- Frames within a case are temporally correlated; frame counts overstate independent evidence.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="JSON file overriding DEFAULT_CONFIG keys")
    ap.add_argument("--no-depth", action="store_true", help="Skip MiDaS; depth factor reported as BLOCKED")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.config:
        deep_update(cfg, json.loads(args.config.read_text(encoding="utf-8")))

    meta_all = load_cases_meta()
    case_map = resolve_case_dirs(REQUESTED_CASES)
    print("Resolved cases:", case_map)

    depth, block_reason = None, None
    if args.no_depth:
        block_reason = "disabled with --no-depth"
    else:
        try:
            depth = DepthEstimator()
        except Exception as e:  # dependency/model unavailable -> BLOCKED, never substituted
            block_reason = f"{type(e).__name__}: {e}"
            print(f"[WARN] MiDaS unavailable, depth factor BLOCKED: {block_reason}")
    depth_status = ("Computed with existing MiDaS_small implementation "
                    "(tracking/E8_midas_depth/depth_features.py)" if depth else f"BLOCKED — {block_reason}")

    all_rows: List[Dict[str, Any]] = []
    prechecks: List[Dict[str, Any]] = []
    for req, cid in case_map.items():
        if cid not in meta_all:
            prechecks.append({"case_id": cid, "requested_as": req, "issues": ["case not in CASES_META"]})
            continue
        rows, pre = process_case(req, cid, meta_all[cid], cfg, depth, block_reason)
        prechecks.append(pre)
        if pre["issues"] and not rows:
            print(f"[{cid}] SKIPPED: {pre['issues']}")
            continue
        write_csv(args.out / cid / "frame_conditions.csv", rows)
        all_rows.extend(rows)
        el = sum(r["eligible"] for r in rows)
        print(f"[{cid}] {meta_all[cid]['seq']}: {len(rows)} frames, {el} eligible")

    write_csv(args.out / "all_cases_frame_conditions.csv", all_rows)
    post = post_validate(all_rows, list(case_map.values()), cfg)
    (args.out / "threshold_configuration.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (args.out / "validation_report.json").write_text(
        json.dumps({"prechecks": prechecks, "post_run": post, "depth_status": depth_status}, indent=2),
        encoding="utf-8")
    (args.out / "classification_summary.md").write_text(
        write_summary(all_rows, prechecks, post, cfg, depth_status), encoding="utf-8")
    print("Post-run validation:", json.dumps(post, indent=2))
    failed = [k for k, v in post.items() if v is False]
    if failed:
        print("[FAIL] validation checks failed:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
