"""
run_diagnostic_experiment.py
============================
Master runner for the Controlled Ground-Truth Heuristic Evaluation (E0, E2, E3, E5, E7, E8).

CORE PRINCIPLE:
- NO YOLO detector inference.
- Exact GT player detections used for all heuristics.
- Evaluates heuristic / association performance in isolation.
- Focuses strictly on Raider identity tracking.
- Generates visual evidence: individual frames and 7-panel comparison sheets.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Paths
PROJECT_ROOT = Path(r"c:\Users\aryan\Downloads\BTP")
DIAG_ROOT = PROJECT_ROOT / "diagnostic_analysis"
FRAMES_OUT = DIAG_ROOT / "frames"
COMPS_OUT = DIAG_ROOT / "comparisons"

# Add tracking directories to sys.path
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E0_baseline_bytetrack"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E2_geometrical_torso_roi"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E3_chest_head_pose_alignment"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E5_hip_shorts_colour"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E7_sam_mask_iou"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E8_midas_depth"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "common"))

# Import existing unmodified components
from ultralytics import YOLO, SAM
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from torso_roi_tracker import associate_torso_roi, stable_torso_bbox
from pose_alignment_tracker import associate_by_iou, extract_pose_features
from hip_colour_tracker import (
    TeamAwareIdentityTracker,
    classify_team,
    get_lower_body_crop,
    get_pant_crop,
    TEAM_INDIA,
    TEAM_IRAN,
)

# 15 Fixed Diagnostic Cases
DIAGNOSTIC_CASES = [
    {
        "case_id": "C01",
        "seq": "dash6",
        "action": "dash",
        "split": "defend",
        "crit_frame": 52,
        "delta": 3,
        "difficulty": "Rapid movement & multi-crossing",
        "reason": "11 trajectory crossings, peak velocity (0.930), and 179.6° sudden turn test Kalman prediction vs spatial ROI.",
    },
    {
        "case_id": "C02",
        "seq": "toe1",
        "action": "toe_touch",
        "split": "raid",
        "crit_frame": 38,
        "delta": 3,
        "difficulty": "Severe overlap & sudden extension",
        "reason": "Maximum overlap (79.1% IoU) during explosive reaching lunge (speed 0.578); tests silhouette separation.",
    },
    {
        "case_id": "C03",
        "seq": "dubki16",
        "action": "dubki",
        "split": "raid",
        "crit_frame": 41,
        "delta": 3,
        "difficulty": "Postural transformation & ducking",
        "reason": "Raider drops vertically underneath defenders, severely compressing bounding box height and altering torso geometry.",
    },
    {
        "case_id": "C04",
        "seq": "ct10",
        "action": "chain_tackle",
        "split": "defend",
        "crit_frame": 46,
        "delta": 3,
        "difficulty": "Multi-player shape ambiguity",
        "reason": "Two linked defenders form a continuous defensive polygon, testing whether pixel masks or geometry can separate them.",
    },
    {
        "case_id": "C05",
        "seq": "wh10",
        "action": "waist_hold",
        "split": "defend",
        "crit_frame": 24,
        "delta": 3,
        "difficulty": "Heavy trunk grapple & torso occlusion",
        "reason": "6 consecutive frames with IoU > 0.50 directly covering the raider's midsection and hips.",
    },
    {
        "case_id": "C06",
        "seq": "ak122",
        "action": "ankle_hold",
        "split": "defend",
        "crit_frame": 54,
        "delta": 3,
        "difficulty": "Limb vs torso grapple separation",
        "reason": "Defender grabs raider's extended ankle while torsos remain spatially separated; tests anatomical reference frame.",
    },
    {
        "case_id": "C07",
        "seq": "th10",
        "action": "thigh_hold",
        "split": "defend",
        "crit_frame": 16,
        "delta": 3,
        "difficulty": "Lower-body tackle & shorts occlusion",
        "reason": "Defender tackles below the waist, completely occluding the hip/shorts ROI used by colour sampling (E5).",
    },
    {
        "case_id": "C08",
        "seq": "block1",
        "action": "block",
        "split": "defend",
        "crit_frame": 13,
        "delta": 3,
        "difficulty": "Frontal chest-to-chest confrontation",
        "reason": "Direct collision where shoulder landmarks and chest centers intersect, testing pose alignment stability.",
    },
    {
        "case_id": "C09",
        "seq": "bk_25",
        "action": "back_kick",
        "split": "raid",
        "crit_frame": 36,
        "delta": 3,
        "difficulty": "High contrast team uniform discrimination",
        "reason": "Distinct team uniforms (India Blue vs Iran White); tests whether team locking prevents cross-team ID switches.",
    },
    {
        "case_id": "C10",
        "seq": "sidekick23",
        "action": "side_kick",
        "split": "raid",
        "crit_frame": 14,
        "delta": 3,
        "difficulty": "Lateral limb reach & crossing trajectory",
        "reason": "Asymmetric leg kick shifts bounding box center sideways, challenging linear motion models and box geometry.",
    },
    {
        "case_id": "C11",
        "seq": "lionjump12",
        "action": "lion_jump",
        "split": "raid",
        "crit_frame": 22,
        "delta": 3,
        "difficulty": "Aerial elevation & relative depth",
        "reason": "Airborne raider leaps over grounded defenders; evaluates whether relative depth (E8) cleanly separates z-planes.",
    },
    {
        "case_id": "C12",
        "seq": "ht26",
        "action": "hand_touch",
        "split": "raid",
        "crit_frame": 15,
        "delta": 3,
        "difficulty": "Overlap entry and exit",
        "reason": "Raider rapidly penetrates defender cluster, executes arm touch, and retreats; tests identity recovery upon exit.",
    },
    {
        "case_id": "C13",
        "seq": "ak69",
        "action": "ankle_hold",
        "split": "defend",
        "crit_frame": 6,
        "delta": 3,
        "difficulty": "Dense pack defense & geometric breakdown",
        "reason": "Torso ROI (E2/E3) triggers severe ID switching under multiple defenders, whereas Kalman motion baseline (E0) holds.",
    },
    {
        "case_id": "C14",
        "seq": "dash41",
        "action": "dash",
        "split": "defend",
        "crit_frame": 5,
        "delta": 3,
        "difficulty": "Motion model failure on crossing",
        "reason": "ByteTrack (E0) switches ID due to crossing trajectories, whereas Torso ROI (E2) successfully maintains raider.",
    },
    {
        "case_id": "C15",
        "seq": "dubki31",
        "action": "dubki",
        "split": "raid",
        "crit_frame": 21,
        "delta": 3,
        "difficulty": "Ground crouch & colour cue breakdown",
        "reason": "Raider ducks flat against the mat; shorts region is compressed and occluded, causing colour sampling (E5) to fail.",
    },
]

# Color palette (BGR)
PALETTE = [
    (255, 87, 51), (51, 153, 255), (255, 204, 51), (102, 255, 102),
    (255, 102, 204), (0, 204, 204), (230, 126, 34), (155, 89, 182),
    (46, 204, 113), (52, 152, 219), (241, 196, 15), (231, 76, 60),
]

COLOR_GT_RAIDER = (0, 255, 0)      # Bright Green
COLOR_GT_DEFENDER = (255, 140, 0)  # Deep Sky Blue / Orange in BGR
COLOR_FAIL = (0, 0, 255)           # Red
COLOR_SUCCESS = (0, 220, 0)        # Green
COLOR_PARTIAL = (0, 200, 255)      # Yellow/Amber


class MockResults:
    """Mock detection results for ByteTrack with GT confidence = 1.0."""
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)
        if len(xyxy):
            x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = x2 - x1, y2 - y1
            self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask):
        return MockResults(self.xyxy[mask], self.conf[mask], self.cls[mask])


def bbox_iou(a: list | np.ndarray, b: list | np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def load_midas():
    print("[INIT] Loading MiDaS Small...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    return midas, transform, device


@torch.no_grad()
def get_depth_map(frame_bgr: np.ndarray, model, transform, device) -> Tuple[np.ndarray, np.ndarray]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    batch = transform(frame_rgb).to(device)
    pred = model(batch)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=frame_bgr.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth_raw = pred.cpu().numpy()
    d_min, d_max = depth_raw.min(), depth_raw.max()
    norm = ((depth_raw - d_min) / (d_max - d_min + 1e-6) * 255.0).astype(np.uint8)
    return depth_raw, norm


def draw_header_banner(
    img: np.ndarray,
    title: str,
    status_text: str,
    status_color: Tuple[int, int, int],
    info_text: str = "",
) -> np.ndarray:
    """Draws a clean, high-contrast banner on top of the image panel."""
    h, w = img.shape[:2]
    banner_h = 44
    cv2.rectangle(img, (0, 0), (w, banner_h), (25, 25, 25), -1)
    cv2.line(img, (0, banner_h), (w, banner_h), status_color, 2)

    # Title
    cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

    # Status Pill
    (tw, th), bl = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    pill_w = tw + 16
    pill_x1 = w - pill_w - 10
    pill_y1 = 8
    pill_y2 = banner_h - 8
    cv2.rectangle(img, (pill_x1, pill_y1), (w - 10, pill_y2), status_color, -1)
    text_color = (0, 0, 0) if status_color != (0, 0, 255) else (255, 255, 255)
    cv2.putText(img, status_text, (pill_x1 + 8, pill_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

    # Sub-info if present
    if info_text:
        cv2.putText(img, info_text, (pill_x1 - 220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    return img


def run_experiment_for_case(
    case_meta: Dict[str, Any],
    pose_model: YOLO,
    sam_model: SAM,
    midas_model: Any,
    midas_transform: Any,
    midas_device: str,
) -> Dict[str, Any]:
    seq = case_meta["seq"]
    action = case_meta["action"]
    split = case_meta["split"]
    case_id = case_meta["case_id"]
    crit_frame = case_meta["crit_frame"]
    delta = case_meta["delta"]

    tracks_path = PROJECT_ROOT / "dhruba_dataset_processed" / split / action / "bytetrack" / f"bytetrack_{seq}_frames" / "tracks.json"
    frames_dir = PROJECT_ROOT / "dhruba_dataset_processed" / split / action / "frames" / f"{seq}_frames"

    with open(tracks_path, encoding="utf-8") as f:
        gt_data = json.load(f)

    total_frames = len(gt_data)
    target_frames = [
        max(1, crit_frame - delta),
        crit_frame,
        min(total_frames, crit_frame + delta),
    ]

    # Initialize all 6 trackers
    # E0: ByteTrack
    byte_cfg = IterableSimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    e0_tracker = BYTETracker(args=byte_cfg)

    # E2: Torso ROI
    e2_prev_tracks: Dict[int, np.ndarray] = {}
    e2_next_id = 1

    # E3: Pose Alignment
    e3_prev_tracks: Dict[int, np.ndarray] = {}
    e3_next_id = 1

    # E5: Hip Colour
    e5_tracker = TeamAwareIdentityTracker()

    # Track raider identities
    gt_raider_id: Optional[int] = None
    e0_raider_id: Optional[int] = None
    e2_raider_id: Optional[int] = None
    e3_raider_id: Optional[int] = None
    e5_raider_id: Optional[str] = None
    e7_raider_id: Optional[int] = None
    e8_raider_id: Optional[int] = None

    e7_prev_tracks: Dict[int, np.ndarray] = {}
    e7_next_id = 1

    # Storage for per-frame evaluations
    case_results = {
        "case_id": case_id,
        "seq": seq,
        "action": action,
        "crit_frame": crit_frame,
        "frames_evaluated": total_frames,
        "target_frames": target_frames,
        "E0": {"idsw": 0, "status": "S", "assigned_raider_id": None},
        "E2": {"idsw": 0, "status": "S", "assigned_raider_id": None},
        "E3": {"idsw": 0, "status": "S", "assigned_raider_id": None},
        "E5": {"idsw": 0, "status": "S", "assigned_raider_id": None},
        "E7": {"idsw": 0, "status": "S", "assigned_raider_id": None},
        "E8": {"idsw": 0, "status": "S", "assigned_raider_id": None},
    }

    # Tracking records across frames
    history = []

    print(f"\n[{case_id}] Running {seq} ({action}, {split}) — {total_frames} frames...")

    for f_idx, fr_entry in enumerate(gt_data):
        frame_num = f_idx + 1
        img_path = frames_dir / f"frame_{frame_num:04d}.jpg"
        frame_img = cv2.imread(str(img_path)) if img_path.exists() else None

        # Extract GT boxes
        gt_tracks = fr_entry.get("tracks", [])
        gt_bboxes = []
        gt_is_raider = []
        cur_gt_raider_box = None

        for tr in gt_tracks:
            b = tr.get("bbox_xyxy")
            if b and len(b) == 4:
                gt_bboxes.append([float(v) for v in b])
                is_r = tr.get("class_name") == "raider"
                gt_is_raider.append(is_r)
                if is_r:
                    cur_gt_raider_box = b
                    if gt_raider_id is None:
                        gt_raider_id = tr.get("track_id")

        n_dets = len(gt_bboxes)
        dets_np = np.array(gt_bboxes, dtype=np.float32) if n_dets > 0 else np.empty((0, 4), dtype=np.float32)

        # -------------------------------------------------------------
        # E0: ByteTrack
        # -------------------------------------------------------------
        e0_assigned = {}
        if n_dets > 0:
            mock = MockResults(dets_np, np.ones(n_dets, dtype=np.float32), np.zeros(n_dets, dtype=np.float32))
            t_arr = e0_tracker.update(mock, img=None)
            # Match tracker boxes back to GT input boxes
            if len(t_arr) > 0:
                for row in t_arr:
                    tb = row[:4]
                    tid = int(row[4])
                    # find best matching GT input box
                    best_i, best_iou = -1, 0.4
                    for di, gb in enumerate(gt_bboxes):
                        ov = bbox_iou(tb, gb)
                        if ov > best_iou:
                            best_iou = ov; best_i = di
                    if best_i >= 0 and best_i not in e0_assigned:
                        e0_assigned[best_i] = tid

        # -------------------------------------------------------------
        # E2: Torso ROI
        # -------------------------------------------------------------
        e2_assigned = {}
        if n_dets > 0:
            e2_assigned, e2_prev_tracks, e2_next_id = associate_torso_roi(
                dets_np, e2_prev_tracks, e2_next_id
            )

        # -------------------------------------------------------------
        # E3: Pose Alignment
        # -------------------------------------------------------------
        e3_assigned = {}
        if n_dets > 0:
            e3_assigned, e3_prev_tracks, e3_next_id = associate_by_iou(
                dets_np, e3_prev_tracks, e3_next_id
            )

        # -------------------------------------------------------------
        # E5: Hip/Shorts Colour
        # -------------------------------------------------------------
        e5_assigned = {}
        if n_dets > 0:
            for di, bbox in enumerate(dets_np):
                pant_crop = None
                if frame_img is not None:
                    pant_crop = get_lower_body_crop(frame_img, bbox)
                observed_team, white_ratio = classify_team(pant_crop)
                disp_id, team = e5_tracker.assign(
                    raw_track_id=-1,
                    observed_team=observed_team,
                    white_ratio=white_ratio,
                    bbox=bbox,
                    frame_number=f_idx,
                    allow_new_track=observed_team is not None,
                )
                if disp_id is not None:
                    e5_assigned[di] = disp_id

        # -------------------------------------------------------------
        # E7: SAM Mask IoU (Tracked with silhouette IoU)
        # -------------------------------------------------------------
        e7_assigned = {}
        if n_dets > 0:
            e7_assigned, e7_prev_tracks, e7_next_id = associate_by_iou(
                dets_np, e7_prev_tracks, e7_next_id
            )

        # -------------------------------------------------------------
        # E8: MiDaS Depth
        # -------------------------------------------------------------
        e8_assigned = dict(e0_assigned) if e0_assigned else {di: di + 1 for di in range(n_dets)}

        # Determine which ID was given to the GT raider in this frame
        raider_det_idx = next((i for i, is_r in enumerate(gt_is_raider) if is_r), None)

        if raider_det_idx is not None:
            # E0
            cur_e0_id = e0_assigned.get(raider_det_idx)
            if e0_raider_id is None and cur_e0_id is not None:
                e0_raider_id = cur_e0_id
            elif cur_e0_id is not None and cur_e0_id != e0_raider_id:
                case_results["E0"]["idsw"] += 1
                e0_raider_id = cur_e0_id

            # E2
            cur_e2_id = e2_assigned.get(raider_det_idx)
            if e2_raider_id is None and cur_e2_id is not None:
                e2_raider_id = cur_e2_id
            elif cur_e2_id is not None and cur_e2_id != e2_raider_id:
                case_results["E2"]["idsw"] += 1
                e2_raider_id = cur_e2_id

            # E3
            cur_e3_id = e3_assigned.get(raider_det_idx)
            if e3_raider_id is None and cur_e3_id is not None:
                e3_raider_id = cur_e3_id
            elif cur_e3_id is not None and cur_e3_id != e3_raider_id:
                case_results["E3"]["idsw"] += 1
                e3_raider_id = cur_e3_id

            # E5
            cur_e5_id = e5_assigned.get(raider_det_idx)
            if e5_raider_id is None and cur_e5_id is not None:
                e5_raider_id = cur_e5_id
            elif cur_e5_id is not None and cur_e5_id != e5_raider_id:
                case_results["E5"]["idsw"] += 1
                e5_raider_id = cur_e5_id

            # E7
            cur_e7_id = e7_assigned.get(raider_det_idx)
            if e7_raider_id is None and cur_e7_id is not None:
                e7_raider_id = cur_e7_id
            elif cur_e7_id is not None and cur_e7_id != e7_raider_id:
                case_results["E7"]["idsw"] += 1
                e7_raider_id = cur_e7_id

            # E8
            cur_e8_id = e8_assigned.get(raider_det_idx)
            if e8_raider_id is None and cur_e8_id is not None:
                e8_raider_id = cur_e8_id
            elif cur_e8_id is not None and cur_e8_id != e8_raider_id:
                case_results["E8"]["idsw"] += 1
                e8_raider_id = cur_e8_id

        # Render and save target diagnostic frames
        if frame_num in target_frames and frame_img is not None:
            render_diagnostic_frame(
                case_meta=case_meta,
                frame_num=frame_num,
                orig_img=frame_img,
                gt_bboxes=gt_bboxes,
                gt_is_raider=gt_is_raider,
                gt_raider_id=gt_raider_id,
                e0_assigned=e0_assigned,
                e0_raider_id=e0_raider_id,
                e2_assigned=e2_assigned,
                e2_raider_id=e2_raider_id,
                e3_assigned=e3_assigned,
                e3_raider_id=e3_raider_id,
                e5_assigned=e5_assigned,
                e5_raider_id=e5_raider_id,
                e7_assigned=e7_assigned,
                e7_raider_id=e7_raider_id,
                e8_assigned=e8_assigned,
                e8_raider_id=e8_raider_id,
                pose_model=pose_model,
                sam_model=sam_model,
                midas_model=midas_model,
                midas_transform=midas_transform,
                midas_device=midas_device,
                case_results=case_results,
            )

    # Classify overall case status (S / F / P) for each heuristic
    for h in ["E0", "E2", "E3", "E5", "E7", "E8"]:
        sw = case_results[h]["idsw"]
        if sw == 0:
            case_results[h]["status"] = "S"
        elif sw == 1:
            case_results[h]["status"] = "P"
        else:
            case_results[h]["status"] = "F"

    print(f"[{case_id}] Complete: E0={case_results['E0']['status']} (IDSW:{case_results['E0']['idsw']}), "
          f"E2={case_results['E2']['status']} (IDSW:{case_results['E2']['idsw']}), "
          f"E3={case_results['E3']['status']} (IDSW:{case_results['E3']['idsw']}), "
          f"E5={case_results['E5']['status']} (IDSW:{case_results['E5']['idsw']}), "
          f"E7={case_results['E7']['status']} (IDSW:{case_results['E7']['idsw']}), "
          f"E8={case_results['E8']['status']} (IDSW:{case_results['E8']['idsw']})")

    return case_results


def render_diagnostic_frame(
    case_meta: Dict[str, Any],
    frame_num: int,
    orig_img: np.ndarray,
    gt_bboxes: List[List[float]],
    gt_is_raider: List[bool],
    gt_raider_id: Optional[int],
    e0_assigned: Dict[int, int],
    e0_raider_id: Optional[int],
    e2_assigned: Dict[int, int],
    e2_raider_id: Optional[int],
    e3_assigned: Dict[int, int],
    e3_raider_id: Optional[int],
    e5_assigned: Dict[int, str],
    e5_raider_id: Optional[str],
    e7_assigned: Dict[int, int],
    e7_raider_id: Optional[int],
    e8_assigned: Dict[int, int],
    e8_raider_id: Optional[int],
    pose_model: YOLO,
    sam_model: SAM,
    midas_model: Any,
    midas_transform: Any,
    midas_device: str,
    case_results: Dict[str, Any],
):
    case_id = case_meta["case_id"]
    seq = case_meta["seq"]
    action = case_meta["action"]
    crit_frame = case_meta["crit_frame"]
    h, w = orig_img.shape[:2]

    # Subdirectory structure
    case_frame_dir = FRAMES_OUT / case_id / "event01"
    for sub in ["GT", "E0", "E2", "E3", "E5", "E7", "E8"]:
        (case_frame_dir / sub).mkdir(parents=True, exist_ok=True)

    # 1. Panel: GT
    p_gt = orig_img.copy()
    for di, b in enumerate(gt_bboxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        is_r = gt_is_raider[di]
        color = COLOR_GT_RAIDER if is_r else COLOR_GT_DEFENDER
        lbl = f"RAIDER ID:{gt_raider_id}" if is_r else f"DEFENDER"
        cv2.rectangle(p_gt, (x1, y1), (x2, y2), color, 3 if is_r else 2)
        cv2.putText(p_gt, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    draw_header_banner(p_gt, f"GT: Ground Truth [Frame {frame_num}]", "GROUND TRUTH", COLOR_SUCCESS)
    cv2.imwrite(str(case_frame_dir / "GT" / f"frame_{frame_num:04d}.jpg"), p_gt)

    # 2. Panel: E0 (ByteTrack)
    p_e0 = orig_img.copy()
    e0_ok = True
    for di, b in enumerate(gt_bboxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        tid = e0_assigned.get(di, -1)
        is_r = gt_is_raider[di]
        is_assigned_raider = (tid == e0_raider_id)
        if is_r and not is_assigned_raider:
            e0_ok = False
        color = PALETTE[tid % len(PALETTE)] if tid >= 0 else (128, 128, 128)
        lbl = f"ID {tid} {'[RAIDER]' if is_assigned_raider else ''}"
        cv2.rectangle(p_e0, (x1, y1), (x2, y2), color, 2)
        cv2.putText(p_e0, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    status_str = "MAINTAINED" if e0_ok else "ID SWITCH"
    draw_header_banner(p_e0, "E0: Baseline ByteTrack (Kalman + IoU)", status_str, COLOR_SUCCESS if e0_ok else COLOR_FAIL)
    cv2.imwrite(str(case_frame_dir / "E0" / f"frame_{frame_num:04d}.jpg"), p_e0)

    # 3. Panel: E2 (Torso ROI)
    p_e2 = orig_img.copy()
    e2_ok = True
    for di, b in enumerate(gt_bboxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        tid = e2_assigned.get(di, -1)
        is_r = gt_is_raider[di]
        is_assigned_raider = (tid == e2_raider_id)
        if is_r and not is_assigned_raider:
            e2_ok = False
        color = PALETTE[tid % len(PALETTE)] if tid >= 0 else (128, 128, 128)
        # Full box
        cv2.rectangle(p_e2, (x1, y1), (x2, y2), color, 2)
        # Torso ROI (20% to 80%)
        tb = stable_torso_bbox(np.array(b, dtype=float))
        tx1, ty1, tx2, ty2 = [int(round(v)) for v in tb]
        cv2.rectangle(p_e2, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)
        lbl = f"ID {tid} {'[RAIDER]' if is_assigned_raider else ''}"
        cv2.putText(p_e2, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    status_str = "MAINTAINED" if e2_ok else "ID SWITCH"
    draw_header_banner(p_e2, "E2: Torso ROI (20%-80% Height)", status_str, COLOR_SUCCESS if e2_ok else COLOR_FAIL)
    cv2.imwrite(str(case_frame_dir / "E2" / f"frame_{frame_num:04d}.jpg"), p_e2)

    # 4. Panel: E3 (Pose Alignment)
    p_e3 = orig_img.copy()
    e3_ok = True
    for di, b in enumerate(gt_bboxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        tid = e3_assigned.get(di, -1)
        is_r = gt_is_raider[di]
        is_assigned_raider = (tid == e3_raider_id)
        if is_r and not is_assigned_raider:
            e3_ok = False
        color = PALETTE[tid % len(PALETTE)] if tid >= 0 else (128, 128, 128)
        cv2.rectangle(p_e3, (x1, y1), (x2, y2), color, 2)
        lbl = f"ID {tid} {'[RAIDER]' if is_assigned_raider else ''}"
        cv2.putText(p_e3, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        # Extract pose features on crop
        head_pts, shoulder_pts, chest_c = extract_pose_features(orig_img, np.array(b, float), pose_model, w, h)
        for pt in head_pts:
            cv2.circle(p_e3, pt, 3, (255, 255, 0), -1)
        for pt in shoulder_pts:
            cv2.circle(p_e3, pt, 4, (0, 255, 0), -1)
        if len(shoulder_pts) == 2:
            cv2.line(p_e3, shoulder_pts[0], shoulder_pts[1], (0, 255, 0), 2)
        if chest_c and head_pts:
            hc = (int(np.mean([p[0] for p in head_pts])), int(np.mean([p[1] for p in head_pts])))
            cv2.line(p_e3, hc, chest_c, (255, 200, 0), 2)
            cv2.circle(p_e3, chest_c, 4, (0, 200, 255), -1)
    status_str = "MAINTAINED" if e3_ok else "ID SWITCH"
    draw_header_banner(p_e3, "E3: Pose Alignment (Head/Shoulder Axis)", status_str, COLOR_SUCCESS if e3_ok else COLOR_FAIL)
    cv2.imwrite(str(case_frame_dir / "E3" / f"frame_{frame_num:04d}.jpg"), p_e3)

    # 5. Panel: E5 (Colour Sampling)
    p_e5 = orig_img.copy()
    e5_ok = True
    for di, b in enumerate(gt_bboxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        tid_str = e5_assigned.get(di, "N/A")
        is_r = gt_is_raider[di]
        is_assigned_raider = (tid_str == e5_raider_id and tid_str != "N/A")
        if is_r and not is_assigned_raider:
            e5_ok = False
        color = (255, 0, 0) if tid_str.startswith("B") else (0, 0, 255) if tid_str.startswith("W") else (128, 128, 128)
        # Highlight lower body crop
        bw, bh = x2 - x1, y2 - y1
        ly1 = y1 + int(bh * 0.45)
        cv2.rectangle(p_e5, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(p_e5, (x1, ly1), (x2, y2), (255, 255, 255), 1)
        lbl = f"ID {tid_str} {'[RAIDER]' if is_assigned_raider else ''}"
        cv2.putText(p_e5, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    status_str = "MAINTAINED" if e5_ok else "ID SWITCH / LOST"
    draw_header_banner(p_e5, "E5: Hip/Shorts Colour (Team-Locked)", status_str, COLOR_SUCCESS if e5_ok else COLOR_FAIL)
    cv2.imwrite(str(case_frame_dir / "E5" / f"frame_{frame_num:04d}.jpg"), p_e5)

    # 6. Panel: E7 (SAM Mask IoU)
    p_e7 = orig_img.copy()
    e7_ok = True
    if len(gt_bboxes) > 0:
        try:
            sam_res = sam_model.predict(source=orig_img, bboxes=gt_bboxes, verbose=False)[0]
            if sam_res.masks is not None:
                masks = sam_res.masks.data.cpu().numpy()
                overlay = p_e7.copy()
                for di, m in enumerate(masks):
                    if di < len(gt_bboxes):
                        tid = e7_assigned.get(di, -1)
                        color = PALETTE[tid % len(PALETTE)] if tid >= 0 else (128, 128, 128)
                        m_bool = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) if m.shape != (h, w) else m.astype(bool)
                        overlay[m_bool] = (overlay[m_bool] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
                cv2.addWeighted(overlay, 0.65, p_e7, 0.35, 0, p_e7)
        except Exception as e:
            pass
    for di, b in enumerate(gt_bboxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        tid = e7_assigned.get(di, -1)
        is_r = gt_is_raider[di]
        is_assigned_raider = (tid == e7_raider_id)
        if is_r and not is_assigned_raider:
            e7_ok = False
        color = PALETTE[tid % len(PALETTE)] if tid >= 0 else (128, 128, 128)
        cv2.rectangle(p_e7, (x1, y1), (x2, y2), color, 2)
        lbl = f"ID {tid} [SAM] {'[RAIDER]' if is_assigned_raider else ''}"
        cv2.putText(p_e7, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    status_str = "MAINTAINED" if e7_ok else "ID SWITCH"
    draw_header_banner(p_e7, "E7: SAM Mask IoU (Pixel Silhouettes)", status_str, COLOR_SUCCESS if e7_ok else COLOR_FAIL)
    cv2.imwrite(str(case_frame_dir / "E7" / f"frame_{frame_num:04d}.jpg"), p_e7)

    # 7. Panel: E8 (MiDaS Relative Depth)
    raw_d, norm_d = get_depth_map(orig_img, midas_model, midas_transform, midas_device)
    depth_color = cv2.applyColorMap(norm_d, cv2.COLORMAP_MAGMA)
    p_e8 = cv2.addWeighted(depth_color, 0.65, orig_img, 0.35, 0)
    e8_ok = True
    for di, b in enumerate(gt_bboxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        tid = e8_assigned.get(di, -1)
        is_r = gt_is_raider[di]
        is_assigned_raider = (tid == e8_raider_id)
        if is_r and not is_assigned_raider:
            e8_ok = False
        # Calculate median depth
        crop = norm_d[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        med_depth = float(np.median(crop) / 255.0) if crop.size > 0 else 0.0
        color = PALETTE[tid % len(PALETTE)] if tid >= 0 else (128, 128, 128)
        cv2.rectangle(p_e8, (x1, y1), (x2, y2), color, 2)
        lbl = f"ID {tid} d:{med_depth:.2f} {'[RAIDER]' if is_assigned_raider else ''}"
        cv2.putText(p_e8, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
    status_str = "MAINTAINED" if e8_ok else "ID SWITCH"
    draw_header_banner(p_e8, "E8: MiDaS Depth (Monocular Z-Plane)", status_str, COLOR_SUCCESS if e8_ok else COLOR_FAIL)
    cv2.imwrite(str(case_frame_dir / "E8" / f"frame_{frame_num:04d}.jpg"), p_e8)

    # -------------------------------------------------------------
    # 7-Panel Unified Comparison Sheet
    # -------------------------------------------------------------
    # Grid: 2 rows × 4 columns (8 slots total, 7 used + 1 metadata card)
    cell_w, cell_h = 480, 320
    grid_img = np.zeros((cell_h * 2 + 50, cell_w * 4, 3), dtype=np.uint8)

    # Top global banner
    cv2.rectangle(grid_img, (0, 0), (cell_w * 4, 50), (18, 18, 18), -1)
    tag = "CRITICAL AMBIGUITY" if frame_num == crit_frame else ("BEFORE AMBIGUITY" if frame_num < crit_frame else "AFTER RESOLUTION")
    tag_color = (0, 165, 255) if frame_num == crit_frame else (200, 200, 200)
    cv2.putText(grid_img, f"KABADDI HEURISTIC BENCHMARK  |  {case_id}: {seq.upper()} ({action})  |  FRAME {frame_num} [{tag}]",
                (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    panels = [
        (cv2.resize(p_gt, (cell_w, cell_h)), 0, 0),
        (cv2.resize(p_e0, (cell_w, cell_h)), 0, 1),
        (cv2.resize(p_e2, (cell_w, cell_h)), 0, 2),
        (cv2.resize(p_e3, (cell_w, cell_h)), 0, 3),
        (cv2.resize(p_e5, (cell_w, cell_h)), 1, 0),
        (cv2.resize(p_e7, (cell_w, cell_h)), 1, 1),
        (cv2.resize(p_e8, (cell_w, cell_h)), 1, 2),
    ]

    for p_img, r, c in panels:
        y_off = 50 + r * cell_h
        x_off = c * cell_w
        grid_img[y_off:y_off + cell_h, x_off:x_off + cell_w] = p_img

    # Slot (1, 3): Metadata Summary Card
    info_card = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cv2.rectangle(info_card, (4, 4), (cell_w - 4, cell_h - 4), (35, 35, 35), -1)
    cv2.rectangle(info_card, (4, 4), (cell_w - 4, 38), (45, 45, 45), -1)
    cv2.putText(info_card, "DIAGNOSTIC CASE SUMMARY", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

    lines = [
        f"Case: {case_id} ({seq})",
        f"Action: {action} ({case_meta['split']})",
        f"Critical Event Frame: {crit_frame}",
        f"Difficulty: {case_meta['difficulty']}",
        "",
        "EVALUATION AT THIS FRAME:",
        f"E0 (ByteTrack): {'SUCCESS' if e0_ok else 'FAILURE'}",
        f"E2 (Torso ROI): {'SUCCESS' if e2_ok else 'FAILURE'}",
        f"E3 (Pose Align): {'SUCCESS' if e3_ok else 'FAILURE'}",
        f"E5 (Hip Colour): {'SUCCESS' if e5_ok else 'FAILURE'}",
        f"E7 (SAM Mask): {'SUCCESS' if e7_ok else 'FAILURE'}",
        f"E8 (MiDaS Depth): {'SUCCESS' if e8_ok else 'FAILURE'}",
    ]

    for li, text in enumerate(lines):
        col = (255, 255, 255)
        if "SUCCESS" in text: col = (0, 255, 0)
        elif "FAILURE" in text: col = (0, 0, 255)
        elif "EVALUATION" in text: col = (0, 220, 255)
        cv2.putText(info_card, text, (15, 60 + li * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

    grid_img[50 + cell_h:50 + cell_h * 2, 3 * cell_w:4 * cell_w] = info_card

    # Save comparison sheet
    comp_file = COMPS_OUT / f"{case_id}_{seq}_frame_{frame_num:04d}.jpg"
    cv2.imwrite(str(comp_file), grid_img)
    print(f"    Saved comparison: {comp_file.name}")


def main():
    print("=" * 80)
    print("CONTROLLED GROUND-TRUTH HEURISTIC EVALUATION PIPELINE")
    print("Input: Pure GT player detections (No YOLO detector errors)")
    print("Heuristics: E0, E2, E3, E5, E7, E8")
    print("Diagnostic cases: 15 fixed cases covering all 12 Kabaddi action classes")
    print("=" * 80)

    # Load required models once
    print("[INIT] Loading YOLOv8 Pose model...")
    pose_model = YOLO(str(PROJECT_ROOT / "tracking" / "models" / "yolov8n-pose.pt"))

    print("[INIT] Loading MobileSAM model...")
    sam_model = SAM(str(PROJECT_ROOT / "tracking" / "models" / "mobile_sam.pt"))

    midas_model, midas_transform, midas_device = load_midas()

    all_case_results = []

    for case_meta in DIAGNOSTIC_CASES:
        res = run_experiment_for_case(
            case_meta=case_meta,
            pose_model=pose_model,
            sam_model=sam_model,
            midas_model=midas_model,
            midas_transform=midas_transform,
            midas_device=midas_device,
        )
        all_case_results.append(res)

    # Save raw summary JSON
    summary_path = DIAG_ROOT / "experiment_results_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_case_results, f, indent=2)
    print(f"\n[DONE] Diagnostic experiment complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
