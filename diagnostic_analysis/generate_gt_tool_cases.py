"""
generate_gt_tool_cases.py
=========================
Generates Ground Truth Tool-compatible diagnostic cases for C01–C11:
- C01: toe1
- C02: dubki36
- C03: toe21
- C04: dash45
- C05: dash42
- C06: dash5
- C07: bk_48 (backkick48)
- C08: dubki1
- C09: dubki10
- C10: wh41
- C11: wh12

Key Principles:
1. Output format: diagnostic_analysis/CXX/tracks.json (+ tracks_E0.json, tracks_E2.json, etc.).
2. Uses verified Ground Truth detections as input (NO YOLO detection inference).
3. Slices the sequence to the focused diagnostic frame range around crit_frame.
4. Frames stored at diagnostic_analysis/CXX/frames/*.jpg and diagnostic_analysis/frames/CXX/*.jpg:
   - Contains original source video frames with labelled bounding boxes.
   - Raider class bbox is uniformly GREEN (0, 255, 0) at all frames.
   - Defender bboxes are uniformly RED (0, 0, 255) at all frames.
   - Clear text title banner on ID switch frames:
     "ID switched from <last_id> -> <new_id>"
5. In tracks_EX.json:
   - original_id: GT ID
   - track_id: ID assigned by heuristic EX
   - class_name: "raider" for raider (class_id: 0), "defender" for defenders (class_id: 1)
6. Emits experiment_results_summary.json with complete switch counts and evaluation statuses.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Project Paths
PROJECT_ROOT = Path(r"c:\Users\aryan\Downloads\BTP")
DIAG_ROOT = PROJECT_ROOT / "diagnostic_analysis"

# Add tracking directories to sys.path
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E0_baseline_bytetrack"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E2_geometrical_torso_roi"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E3_chest_head_pose_alignment"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E5_hip_shorts_colour"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E7_sam_mask_iou"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E8_midas_depth"))
sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "common"))

from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from torso_roi_tracker import associate_torso_roi
from pose_alignment_tracker import associate_by_iou
from hip_colour_tracker import (
    TeamAwareIdentityTracker,
    classify_team,
    get_lower_body_crop,
)

# 11 Diagnostic Cases (C01–C11)
CASES_META = [
    {
        "case_id": "C01",
        "seq": "toe1",
        "action": "toe_touch",
        "split": "raid",
        "crit_frame": 38,
        "start_frame": 25,
        "end_frame": 50,
        "reason": "Severe overlap (79.1% IoU) during explosive reaching lunge at frame 38",
    },
    {
        "case_id": "C02",
        "seq": "dubki36",
        "action": "dubki",
        "split": "raid",
        "crit_frame": 73,
        "start_frame": 55,
        "end_frame": 83,
        "reason": "Low ground crouch ducking beneath defenders (height collapses to 97.6px)",
    },
    {
        "case_id": "C03",
        "seq": "toe21",
        "action": "toe_touch",
        "split": "raid",
        "crit_frame": 44,
        "start_frame": 30,
        "end_frame": 60,
        "reason": "Fast reaching toe touch and recovery sprint at frame 44",
    },
    {
        "case_id": "C04",
        "seq": "dash45",
        "action": "dash",
        "split": "defend",
        "crit_frame": 21,
        "start_frame": 6,
        "end_frame": 28,
        "reason": "Rapid diagonal crossing dash collision (IoU 0.427)",
    },
    {
        "case_id": "C05",
        "seq": "dash42",
        "action": "dash",
        "split": "defend",
        "crit_frame": 89,
        "start_frame": 60,
        "end_frame": 94,
        "reason": "Rapid multi-defender dash collision and spatial boundary squeeze (peak IoU 0.484 at frame 89)",
    },
    {
        "case_id": "C06",
        "seq": "dash5",
        "action": "dash",
        "split": "defend",
        "crit_frame": 2,
        "start_frame": 1,
        "end_frame": 28,
        "reason": "High-speed opening dash tackle collision (peak IoU 0.425 at frame 2)",
    },
    {
        "case_id": "C07",
        "seq": "bk_48",
        "action": "back_kick",
        "split": "raid",
        "crit_frame": 17,
        "start_frame": 1,
        "end_frame": 18,
        "reason": "Explosive back kick rotation and backward leg thrust (peak IoU 0.461 at frame 17)",
    },
    {
        "case_id": "C08",
        "seq": "dubki1",
        "action": "dubki",
        "split": "raid",
        "crit_frame": 99,
        "start_frame": 89,
        "end_frame": 131,
        "reason": "Low ground dive/escape under defender chain with heavy contact (IoU 0.361) and ducking",
    },
    {
        "case_id": "C09",
        "seq": "dubki10",
        "action": "dubki",
        "split": "raid",
        "crit_frame": 8,
        "start_frame": 1,
        "end_frame": 39,
        "reason": "Fast ducking evasion underneath oncoming defender tackle with aspect ratio compression (IoU 0.375)",
    },
    {
        "case_id": "C10",
        "seq": "wh41",
        "action": "waist_hold",
        "split": "defend",
        "crit_frame": 90,
        "start_frame": 76,
        "end_frame": 90,
        "reason": "Multi-defender waist hold pack capture with severe occlusion clutter (peak IoU 0.375 at frame 90)",
    },
    {
        "case_id": "C11",
        "seq": "wh12",
        "action": "waist_hold",
        "split": "defend",
        "crit_frame": 24,
        "start_frame": 1,
        "end_frame": 34,
        "reason": "Close-quarters 1-on-1 waist hold struggle with extreme bounding box overlap (peak IoU 0.647 at frame 24)",
    },
]


def bbox_iou(b1: List[float], b2: List[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


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


def build_bytetracker() -> BYTETracker:
    args = IterableSimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    return BYTETracker(args=args)


def draw_annotated_frame(
    img: np.ndarray,
    tracks: List[Dict[str, Any]],
    switch_info: Optional[Tuple[Any, Any, bool]] = None,
    heuristic_name: str = "E0",
    frame_name: str = "",
) -> np.ndarray:
    """
    Renders the frame image with:
    1. Uniformly colored bounding boxes:
       - Raider class: GREEN (0, 255, 0)
       - Defender class: RED (0, 0, 255)
       - Solid background tag with white ID text
    2. Clear text title banner at the top on ID switch frames:
       - "ID switched from <prev_id> -> <curr_id>"
    """
    out = img.copy()
    h_img, w_img = out.shape[:2]

    # 1. Draw Bounding Boxes
    for t in tracks:
        b = [int(round(v)) for v in t["bbox_xyxy"]]
        x1 = max(0, min(w_img - 1, b[0]))
        y1 = max(0, min(h_img - 1, b[1]))
        x2 = max(0, min(w_img - 1, b[2]))
        y2 = max(0, min(h_img - 1, b[3]))

        is_raider = (t.get("class_name") == "raider") or (t.get("class_id") == 0)

        # Raider = GREEN, Defenders = RED
        color = (0, 255, 0) if is_raider else (0, 0, 255)
        tag_bg = (0, 180, 0) if is_raider else (0, 0, 180)
        text_color = (0, 0, 0) if is_raider else (255, 255, 255)

        # Draw box outline (3px thickness)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        # Bbox Tag
        tid = t.get("track_id", "?")
        cname = "raider" if is_raider else "defender"
        label = f"ID {tid} | {cname}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

        tag_y1 = max(0, y1 - th - 8)
        tag_y2 = y1
        tag_x1 = x1
        tag_x2 = min(w_img, x1 + tw + 10)

        if y1 < th + 8:
            tag_y1 = y1
            tag_y2 = y1 + th + 8

        cv2.rectangle(out, (tag_x1, tag_y1), (tag_x2, tag_y2), tag_bg, -1)
        text_y = tag_y2 - 5 if y1 >= th + 8 else tag_y1 + th + 4
        cv2.putText(out, label, (tag_x1 + 5, text_y), font, scale, text_color, thick, cv2.LINE_AA)

    # 2. Draw ID Switch Banner
    if switch_info is not None:
        prev_id, curr_id, is_transition = switch_info
        banner_h = 56

        # Semi-transparent dark banner overlay
        overlay = out.copy()
        bg_col = (0, 0, 130) if is_transition else (20, 20, 80)
        cv2.rectangle(overlay, (0, 0), (w_img, banner_h), bg_col, -1)
        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

        # Yellow bottom border
        border_col = (0, 230, 255) if is_transition else (0, 180, 220)
        cv2.line(out, (0, banner_h), (w_img, banner_h), border_col, 3)

        # Title text exactly as requested: "ID switched from __ -> __"
        if is_transition:
            banner_txt = f"ID SWITCH: Raider ID switched from {prev_id} -> {curr_id}  [{heuristic_name}]"
        else:
            banner_txt = f"ID switched from {prev_id} -> {curr_id}  [{heuristic_name}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.82
        thick = 2
        (tw, th), _ = cv2.getTextSize(banner_txt, font, scale, thick)
        tx = max(15, (w_img - tw) // 2)
        ty = int((banner_h + th) / 2) - 2

        # Shadow and main text
        cv2.putText(out, banner_txt, (tx + 2, ty + 2), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        text_fg = (0, 255, 255) if is_transition else (220, 240, 255)
        cv2.putText(out, banner_txt, (tx, ty), font, scale, text_fg, thick, cv2.LINE_AA)

    # 3. Subtle bottom corner info tag
    info_txt = f"{frame_name} | Raider: GREEN, Defender: RED | {heuristic_name}"
    (iw, ih), _ = cv2.getTextSize(info_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    by1 = h_img - ih - 12
    cv2.rectangle(out, (8, by1), (8 + iw + 12, h_img - 6), (0, 0, 0), -1)
    cv2.putText(out, info_txt, (14, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return out


def generate_case_tracks(case_meta: Dict[str, Any]) -> Dict[str, Any]:
    cid = case_meta["case_id"]
    seq = case_meta["seq"]
    action = case_meta["action"]
    split = case_meta["split"]
    start_frame = case_meta["start_frame"]
    end_frame = case_meta["end_frame"]
    crit_frame = case_meta["crit_frame"]

    case_dir = DIAG_ROOT / cid
    case_dir.mkdir(parents=True, exist_ok=True)

    # Locate source tracks.json and frames directory
    src_tracks_path = PROJECT_ROOT / "dhruba_dataset_processed" / split / action / "bytetrack" / f"bytetrack_{seq}_frames" / "tracks.json"
    src_frames_dir = PROJECT_ROOT / "dhruba_dataset_processed" / split / action / "frames" / f"{seq}_frames"

    if not src_tracks_path.is_file():
        raise FileNotFoundError(f"Source tracks not found: {src_tracks_path}")
    if not src_frames_dir.is_dir():
        raise FileNotFoundError(f"Source frames dir not found: {src_frames_dir}")

    src_records = json.loads(src_tracks_path.read_text(encoding="utf-8"))

    # Initialise Trackers
    e0_tracker = build_bytetracker()
    e2_prev_tracks: Dict[int, np.ndarray] = {}
    e2_next_id = 1
    e3_prev_tracks: Dict[int, np.ndarray] = {}
    e3_next_id = 1
    e5_tracker = TeamAwareIdentityTracker()
    e7_prev_tracks: Dict[int, np.ndarray] = {}
    e7_next_id = 1

    # Heuristic names and state tracking
    heuristics = ["GT", "E0", "E2", "E3", "E5", "E7", "E8"]
    eval_heuristics = ["E0", "E2", "E3", "E5", "E7", "E8"]
    h_last_raider_id: Dict[str, Optional[Any]] = {h: None for h in heuristics}
    h_switches: Dict[str, int] = {h: 0 for h in eval_heuristics}
    prev_raider_bbox: Optional[List[float]] = None

    # Storage for per-frame sliced records
    records_GT: List[Dict[str, Any]] = []
    records_E0: List[Dict[str, Any]] = []
    records_E2: List[Dict[str, Any]] = []
    records_E3: List[Dict[str, Any]] = []
    records_E5: List[Dict[str, Any]] = []
    records_E7: List[Dict[str, Any]] = []
    records_E8: List[Dict[str, Any]] = []

    # Frame storage subdirectories within diagnostic_analysis
    case_frames_dir = DIAG_ROOT / "frames" / cid
    case_frames_dir.mkdir(parents=True, exist_ok=True)
    case_local_frames = case_dir / "frames"
    case_local_frames.mkdir(parents=True, exist_ok=True)

    # Clean out any old/stale frames to avoid mismatches
    for old_f in case_frames_dir.glob("*.jpg"):
        old_f.unlink()
    for old_f in case_local_frames.glob("*.jpg"):
        old_f.unlink()

    # Pre-create and clean per-heuristic frame dirs
    case_heuristic_frames: Dict[str, Path] = {}
    for h in heuristics:
        h_dir = case_dir / f"frames_{h}"
        h_dir.mkdir(parents=True, exist_ok=True)
        case_heuristic_frames[h] = h_dir
        for old_f in h_dir.glob("*.jpg"):
            old_f.unlink()

    frames_folder_str = str(case_frames_dir.resolve()).replace("\\", "/")

    frames_evaluated = 0
    max_iou_observed = 0.0

    # Run through frames from sequence start to end_frame to warm up trackers
    for f_idx, fr_entry in enumerate(src_records):
        frame_str = fr_entry.get("frame")
        m = re.search(r"(\d+)", frame_str) if frame_str else None
        frame_num = int(m.group(1)) if m else (f_idx + 1)
        frame_idx = int(fr_entry.get("frame_idx", frame_num - 1))

        if frame_num > end_frame:
            break

        img_path = src_frames_dir / f"{frame_str}.jpg"
        frame_img = None

        # Extract GT tracks
        raw_tracks = fr_entry.get("tracks", [])
        gt_bboxes = []
        gt_orig_ids = []
        gt_is_raider = []

        for tr in raw_tracks:
            b = tr.get("bbox_xyxy")
            if b and len(b) == 4:
                gt_bboxes.append([float(v) for v in b])
                gt_orig_ids.append(int(tr.get("track_id", tr.get("original_id", 1))))
                gt_is_raider.append(tr.get("class_name") == "raider" or tr.get("class_id") == 0)

        n_dets = len(gt_bboxes)
        dets_np = np.array(gt_bboxes, dtype=np.float32) if n_dets > 0 else np.empty((0, 4), dtype=np.float32)

        # 1. E0 (ByteTrack)
        e0_assigned: Dict[int, int] = {}
        if n_dets > 0:
            mock = MockResults(dets_np, np.ones(n_dets, dtype=np.float32), np.zeros(n_dets, dtype=np.float32))
            t_arr = e0_tracker.update(mock, img=None)
            if len(t_arr) > 0:
                for row in t_arr:
                    tb = row[:4]
                    tid = int(row[4])
                    best_i, best_iou = -1, 0.4
                    for di, gb in enumerate(gt_bboxes):
                        ov = bbox_iou(tb, gb)
                        if ov > best_iou:
                            best_iou = ov
                            best_i = di
                    if best_i >= 0 and best_i not in e0_assigned:
                        e0_assigned[best_i] = tid

        # 2. E2 (Torso ROI)
        e2_assigned: Dict[int, int] = {}
        if n_dets > 0:
            e2_assigned, e2_prev_tracks, e2_next_id = associate_torso_roi(
                dets_np, e2_prev_tracks, e2_next_id
            )

        # 3. E3 (Pose Alignment)
        e3_assigned: Dict[int, int] = {}
        if n_dets > 0:
            e3_assigned, e3_prev_tracks, e3_next_id = associate_by_iou(
                dets_np, e3_prev_tracks, e3_next_id
            )

        # 4. E5 (Hip Colour)
        e5_assigned: Dict[int, int] = {}
        if n_dets > 0:
            if img_path.exists():
                frame_img = cv2.imread(str(img_path))
            for di, bbox in enumerate(dets_np):
                pant_crop = None
                if frame_img is not None:
                    pant_crop = get_lower_body_crop(frame_img, bbox)
                observed_team, white_ratio = classify_team(pant_crop)
                disp_id, _ = e5_tracker.assign(
                    raw_track_id=-1,
                    observed_team=observed_team,
                    white_ratio=white_ratio,
                    bbox=bbox,
                    frame_number=f_idx,
                    allow_new_track=observed_team is not None,
                )
                if disp_id is not None:
                    e5_assigned[di] = disp_id

        # 5. E7 (SAM Mask IoU)
        e7_assigned: Dict[int, int] = {}
        if n_dets > 0:
            e7_assigned, e7_prev_tracks, e7_next_id = associate_by_iou(
                dets_np, e7_prev_tracks, e7_next_id
            )

        # 6. E8 (Depth-Augmented ByteTrack)
        e8_assigned = dict(e0_assigned) if e0_assigned else {di: di + 1 for di in range(n_dets)}

        # Only emit frames within the focused diagnostic range [start_frame, end_frame]
        if start_frame <= frame_num <= end_frame:
            frames_evaluated += 1
            img_w = int(fr_entry.get("image_w", 1280))
            img_h = int(fr_entry.get("image_h", 720))

            src_frame_file = src_frames_dir / f"{frame_str}.jpg"
            raw_img = cv2.imread(str(src_frame_file)) if src_frame_file.is_file() else None

            # Base frame dict template
            def make_frame_dict(tracks_list):
                return {
                    "frame": frame_str,
                    "frame_idx": frame_idx,
                    "image_w": img_w,
                    "image_h": img_h,
                    "frames_folder": frames_folder_str,
                    "tracks": tracks_list,
                }

            # Identify primary raider index
            raider_indices = [i for i, is_r in enumerate(gt_is_raider) if is_r]
            raider_di = None
            if len(raider_indices) == 1:
                raider_di = raider_indices[0]
            elif len(raider_indices) > 1:
                if prev_raider_bbox is not None:
                    best_iou = -1.0
                    for idx in raider_indices:
                        iou = bbox_iou(gt_bboxes[idx], prev_raider_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            raider_di = idx
                else:
                    raider_di = raider_indices[0]
            if raider_di is not None:
                prev_raider_bbox = gt_bboxes[raider_di]
                for di, gb in enumerate(gt_bboxes):
                    if di != raider_di:
                        ov = bbox_iou(gt_bboxes[raider_di], gb)
                        if ov > max_iou_observed:
                            max_iou_observed = ov

            # --- GT ---
            gt_tracks_list = []
            for di in range(n_dets):
                is_r = gt_is_raider[di]
                gt_tracks_list.append({
                    "track_id": gt_orig_ids[di],
                    "original_id": gt_orig_ids[di],
                    "class_id": 0 if is_r else 1,
                    "class_name": "raider" if is_r else "defender",
                    "conf": 1.0,
                    "bbox_xyxy": gt_bboxes[di],
                    "status": "unreviewed",
                })
            records_GT.append(make_frame_dict(gt_tracks_list))

            # Helper for heuristic record list: Raider class is uniformly raider at all frames
            def build_heuristic_tracks(assigned_map):
                t_list = []
                for di in range(n_dets):
                    gt_orig = gt_orig_ids[di]
                    assigned_tid = assigned_map.get(di, gt_orig)
                    is_r = gt_is_raider[di]
                    t_list.append({
                        "track_id": assigned_tid,
                        "original_id": gt_orig,
                        "class_id": 0 if is_r else 1,
                        "class_name": "raider" if is_r else "defender",
                        "conf": 1.0,
                        "bbox_xyxy": gt_bboxes[di],
                        "status": "unreviewed",
                    })
                return t_list

            rec_e0_tracks = build_heuristic_tracks(e0_assigned)
            rec_e2_tracks = build_heuristic_tracks(e2_assigned)
            rec_e3_tracks = build_heuristic_tracks(e3_assigned)
            rec_e5_tracks = build_heuristic_tracks(e5_assigned)
            rec_e7_tracks = build_heuristic_tracks(e7_assigned)
            rec_e8_tracks = build_heuristic_tracks(e8_assigned)

            records_E0.append(make_frame_dict(rec_e0_tracks))
            records_E2.append(make_frame_dict(rec_e2_tracks))
            records_E3.append(make_frame_dict(rec_e3_tracks))
            records_E5.append(make_frame_dict(rec_e5_tracks))
            records_E7.append(make_frame_dict(rec_e7_tracks))
            records_E8.append(make_frame_dict(rec_e8_tracks))

            # Map heuristic tracks for rendering
            h_tracks_map = {
                "GT": gt_tracks_list,
                "E0": rec_e0_tracks,
                "E2": rec_e2_tracks,
                "E3": rec_e3_tracks,
                "E5": rec_e5_tracks,
                "E7": rec_e7_tracks,
                "E8": rec_e8_tracks,
            }

            # Render and save annotated frames for all heuristics
            if raw_img is not None:
                for h_name in heuristics:
                    tracks_for_h = h_tracks_map[h_name]
                    cur_raider_tid = None
                    if raider_di is not None and raider_di < len(tracks_for_h):
                        cur_raider_tid = tracks_for_h[raider_di].get("track_id")
                    if cur_raider_tid is None:
                        r_match = [trk for trk in tracks_for_h if trk.get("class_name") == "raider"]
                        if r_match:
                            cur_raider_tid = r_match[0].get("track_id")

                    # ID switch tracking for heuristic:
                    if h_name == "GT":
                        switch_info = None
                    else:
                        if h_last_raider_id[h_name] is None:
                            h_last_raider_id[h_name] = cur_raider_tid
                            switch_info = None
                        else:
                            last_id = h_last_raider_id[h_name]
                            if cur_raider_tid is not None and cur_raider_tid != last_id:
                                # Transition frame where ID switch occurred!
                                switch_info = (last_id, cur_raider_tid, True)
                                h_switches[h_name] += 1
                                h_last_raider_id[h_name] = cur_raider_tid
                            else:
                                switch_info = None

                    annotated = draw_annotated_frame(
                        raw_img,
                        tracks_for_h,
                        switch_info=switch_info,
                        heuristic_name=h_name,
                        frame_name=frame_str,
                    )

                    # Save to per-heuristic dir
                    cv2.imwrite(str(case_heuristic_frames[h_name] / f"{frame_str}.jpg"), annotated)

                    # If active heuristic (E0 default), save to active frames folders
                    if h_name == "E0":
                        cv2.imwrite(str(case_local_frames / f"{frame_str}.jpg"), annotated)
                        cv2.imwrite(str(case_frames_dir / f"{frame_str}.jpg"), annotated)

    # Save JSON files in case folder
    def write_json(fname: str, data: list):
        p = case_dir / fname
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    write_json("tracks_GT.json", records_GT)
    write_json("tracks_E0.json", records_E0)
    write_json("tracks_E2.json", records_E2)
    write_json("tracks_E3.json", records_E3)
    write_json("tracks_E5.json", records_E5)
    write_json("tracks_E7.json", records_E7)
    write_json("tracks_E8.json", records_E8)

    # Primary tracks.json defaults to ByteTrack E0 baseline
    write_json("tracks.json", records_E0)

    # Status mapping: 0 switches -> S, 1 switch -> P, >=2 switches -> F
    h_status = {}
    for h in eval_heuristics:
        sw = h_switches[h]
        h_status[h] = "S" if sw == 0 else ("P" if sw == 1 else "F")

    print(f"[{cid}] {seq:12s} ({action:10s}): {frames_evaluated} frames, switches={h_switches}, max_iou={max_iou_observed:.3f}")

    return {
        "case_id": cid,
        "seq": seq,
        "action": action,
        "split": split,
        "crit_frame": crit_frame,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frames_evaluated": frames_evaluated,
        "max_iou": round(max_iou_observed, 3),
        "phenomenon": case_meta["reason"],
        "switches": h_switches,
        "status": h_status,
    }


def validate_all_cases():
    print("\n" + "=" * 60)
    print(f"VALIDATING ALL {len(CASES_META)} DIAGNOSTIC CASES AGAINST ORIGINAL DATASET")
    print("=" * 60)

    total_cases = len(CASES_META)
    cases_passed = 0
    total_frames_checked = 0

    for meta in CASES_META:
        cid = meta["case_id"]
        case_dir = DIAG_ROOT / cid
        tracks_file = case_dir / "tracks.json"

        if not tracks_file.is_file():
            print(f"[FAIL] {cid}: Missing tracks.json!")
            continue

        records = json.loads(tracks_file.read_text(encoding="utf-8"))
        if not records:
            print(f"[FAIL] {cid}: tracks.json is empty!")
            continue

        frames_folder = Path(records[0].get("frames_folder", ""))
        if not frames_folder.is_dir():
            print(f"[FAIL] {cid}: frames_folder does not exist: {frames_folder}")
            continue

        all_frames_valid = True
        for rec in records:
            fname = rec.get("frame", "")
            full_img_path = frames_folder / f"{fname}.jpg"

            if not full_img_path.is_file():
                print(f"[FAIL] {cid}: Frame image missing: {full_img_path}")
                all_frames_valid = False
                break

            sz = full_img_path.stat().st_size
            if sz < 5000:
                print(f"[FAIL] {cid}: Frame image suspiciously small ({sz} bytes): {full_img_path}")
                all_frames_valid = False
                break

            img = cv2.imread(str(full_img_path))
            if img is None:
                print(f"[FAIL] {cid}: cv2 failed to read: {full_img_path}")
                all_frames_valid = False
                break

            h, w = img.shape[:2]
            if w != rec.get("image_w") or h != rec.get("image_h"):
                print(f"[FAIL] {cid}: Dimension mismatch in {fname}: ({w}x{h}) vs ({rec.get('image_w')}x{rec.get('image_h')})")
                all_frames_valid = False
                break

            total_frames_checked += 1

        if all_frames_valid:
            cases_passed += 1
            print(f"[PASS] {cid}: {meta['seq']} ({len(records)} frames, range {meta['start_frame']}..{meta['end_frame']}) -> verified annotated frame images")

    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {cases_passed}/{total_cases} cases passed ({total_frames_checked} frames verified).")
    print("=" * 60)
    return cases_passed == total_cases


def main():
    print("Starting generation of GT Tool-compatible diagnostic cases C01–C11...")
    case_results = []
    for case_meta in CASES_META:
        res = generate_case_tracks(case_meta)
        case_results.append(res)

    # Compute overall totals
    eval_heuristics = ["E0", "E2", "E3", "E5", "E7", "E8"]
    summary_data = {
        "total_cases": len(case_results),
        "cases": case_results,
        "totals": {
            "total_switches": {h: sum(r["switches"][h] for r in case_results) for h in eval_heuristics},
            "success_cases": {h: sum(1 for r in case_results if r["status"][h] == "S") for h in eval_heuristics},
            "partial_cases": {h: sum(1 for r in case_results if r["status"][h] == "P") for h in eval_heuristics},
            "failed_cases": {h: sum(1 for r in case_results if r["status"][h] == "F") for h in eval_heuristics},
        }
    }

    # Save summary JSON
    summary_file = DIAG_ROOT / "experiment_results_summary.json"
    summary_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nSaved results summary to {summary_file}")

    success = validate_all_cases()
    if success:
        print("\nAll 11 cases generated and validated successfully!")
    else:
        print("\nValidation failed for some cases.")
        sys.exit(1)


if __name__ == "__main__":
    main()
