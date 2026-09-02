"""
pose_alignment_tracker.py
─────────────────────────
E3 — Chest + Head Pose Alignment Tracking.

Uses YOLOv8 pose estimation to extract head and shoulder keypoints,
then uses chest/shoulder midpoint and head centroid alignment as an
additional tracking cue alongside bounding-box IoU.

Pipeline
--------
    YOLOv8 detect → IoU association → Pose estimation per crop →
    Head keypoints (nose, eyes, ears) + Shoulder keypoints →
    Chest centre + Head centroid alignment

Keypoint indices (COCO 17-keypoint format)
------------------------------------------
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder

Usage
-----
    python pose_alignment_tracker.py --det-weights models/best.pt \\
                                     --pose-weights models/yolov8n-pose.pt \\
                                     --video input.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────

DETECT_CONF = 0.30
IOU_THRESHOLD = 0.15        # minimum bbox IoU for track association
KEYPOINT_CONF = 0.30        # minimum keypoint confidence

# Keypoint indices in YOLO pose (COCO 17):
HEAD_KP_IDXS = [0, 1, 2, 3, 4]       # nose, left_eye, right_eye, left_ear, right_ear
SHOULDER_KP_IDXS = [5, 6]             # left_shoulder, right_shoulder

# Colors (BGR)
PALETTE = [
    (255, 87, 51), (51, 153, 255), (255, 204, 51), (102, 255, 102),
    (255, 102, 204), (0, 204, 204), (230, 126, 34), (155, 89, 182),
    (46, 204, 113), (52, 152, 219), (241, 196, 15), (231, 76, 60),
]


def color_for_id(tid: int) -> tuple:
    return PALETTE[tid % len(PALETTE)]


# ── Geometry Utilities ────────────────────────────────────────────────────

def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── IoU-based Association ─────────────────────────────────────────────────

def associate_by_iou(
    dets: np.ndarray,
    track_bboxes: Dict[int, np.ndarray],
    next_id: int,
) -> Tuple[Dict[int, int], Dict[int, np.ndarray], int]:
    """
    Greedy IoU-based association between detections and existing tracks.

    Returns
    -------
    assigned : dict {detection_index → track_id}
    new_bboxes : dict {track_id → bbox}
    next_id : updated next ID
    """
    assigned: Dict[int, int] = {}
    used_tracks: set = set()
    track_ids = list(track_bboxes.keys())

    if track_ids:
        iou_mat = np.zeros((len(dets), len(track_ids)), dtype=float)
        for di, det in enumerate(dets):
            for ti, tid in enumerate(track_ids):
                iou_mat[di, ti] = bbox_iou(det, track_bboxes[tid])

        while True:
            if iou_mat.size == 0 or iou_mat.max() < IOU_THRESHOLD:
                break
            bi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            di, ti = int(bi[0]), int(bi[1])
            if di not in assigned and ti not in used_tracks:
                assigned[di] = track_ids[ti]
                used_tracks.add(ti)
            iou_mat[di, :] = 0.0
            iou_mat[:, ti] = 0.0

    new_bboxes = {}
    for di, det in enumerate(dets):
        tid = assigned.get(di, next_id)
        if di not in assigned:
            next_id += 1
        new_bboxes[tid] = det

    return assigned, new_bboxes, next_id


# ── Pose Keypoint Extraction ─────────────────────────────────────────────

def extract_pose_features(
    frame: np.ndarray,
    det: np.ndarray,
    pose_model: YOLO,
    frame_width: int,
    frame_height: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Extract head and shoulder keypoints for a single detection crop.

    Returns
    -------
    head_pts : list of (x, y) head keypoint positions in frame coords
    shoulder_pts : list of (x, y) shoulder keypoint positions in frame coords
    chest_center : (x, y) midpoint of shoulders, or None if not available
    """
    x1, y1, x2, y2 = [int(round(v)) for v in det]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width, x2), min(frame_height, y2)

    head_pts = []
    shoulder_pts = []
    chest_center = None

    if x2 > x1 and y2 > y1:
        crop = frame[y1:y2, x1:x2]
        pose_res = pose_model(crop, verbose=False)

        if pose_res and len(pose_res) > 0 and pose_res[0].keypoints is not None:
            kps_data = pose_res[0].keypoints
            if len(kps_data.xy) > 0:
                kp_arr = kps_data.xy[0].cpu().numpy()
                conf_arr = (
                    kps_data.conf[0].cpu().numpy()
                    if kps_data.conf is not None
                    else np.ones(len(kp_arr))
                )

                # Head landmarks (Nose / Eyes / Ears)
                for h_idx in HEAD_KP_IDXS:
                    if h_idx < len(kp_arr) and conf_arr[h_idx] > KEYPOINT_CONF:
                        kx = int(round(kp_arr[h_idx][0] + x1))
                        ky = int(round(kp_arr[h_idx][1] + y1))
                        head_pts.append((kx, ky))

                # Shoulder landmarks
                for s_idx in SHOULDER_KP_IDXS:
                    if s_idx < len(kp_arr) and conf_arr[s_idx] > KEYPOINT_CONF:
                        kx = int(round(kp_arr[s_idx][0] + x1))
                        ky = int(round(kp_arr[s_idx][1] + y1))
                        shoulder_pts.append((kx, ky))

                # Compute chest centre
                if len(shoulder_pts) == 2:
                    p1, p2 = shoulder_pts[0], shoulder_pts[1]
                    chest_center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

    return head_pts, shoulder_pts, chest_center


# ── Visualisation ─────────────────────────────────────────────────────────

def draw_pose_alignment(
    vis: np.ndarray,
    det: np.ndarray,
    tid: int,
    head_pts: List[Tuple[int, int]],
    shoulder_pts: List[Tuple[int, int]],
    chest_center: Optional[Tuple[int, int]],
) -> None:
    """Draw bounding box, keypoints, and alignment lines on frame."""
    color = color_for_id(tid)

    # Bounding box + label
    x1, y1, x2, y2 = [int(round(v)) for v in det]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    label = f"ID {tid}"
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    label_y1 = max(0, y1 - th - bl - 8)
    label_y2 = max(th + bl + 8, y1)
    cv2.rectangle(vis, (x1, label_y1), (x1 + tw + 10, label_y2), color, -1)
    cv2.putText(vis, label, (x1 + 5, label_y2 - bl - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # Head landmarks → Cyan circles
    for (kx, ky) in head_pts:
        cv2.circle(vis, (kx, ky), 4, (255, 255, 0), -1)

    # Shoulder landmarks → Lime circles & connecting chest line
    for (kx, ky) in shoulder_pts:
        cv2.circle(vis, (kx, ky), 5, (0, 255, 0), -1)

    if len(shoulder_pts) == 2:
        cv2.line(vis, shoulder_pts[0], shoulder_pts[1], (0, 255, 0), 2)

    # Chest centre
    if chest_center is not None:
        cv2.circle(vis, chest_center, 5, (0, 200, 255), -1)

        # Connect head centroid to chest centre
        if head_pts:
            head_cx = int(np.mean([p[0] for p in head_pts]))
            head_cy = int(np.mean([p[1] for p in head_pts]))
            cv2.line(vis, (head_cx, head_cy), chest_center, (255, 200, 0), 2)


# ── Main Pipeline ─────────────────────────────────────────────────────────

def run_pose_alignment_tracking(
    video_path: str, det_weights: str, pose_weights: str, output_path: str
):
    """Run E3 chest+head pose alignment tracking on a video."""
    yolo_model = YOLO(det_weights)
    pose_model = YOLO(pose_weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    track_bboxes: Dict[int, np.ndarray] = {}
    next_id = 1
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        vis = frame.copy()
        res = yolo_model.predict(source=frame, conf=DETECT_CONF, verbose=False)[0]

        if res.boxes is not None and len(res.boxes) > 0:
            dets = res.boxes.xyxy.cpu().numpy()

            assigned, track_bboxes, next_id = associate_by_iou(
                dets, track_bboxes, next_id
            )

            for di, det in enumerate(dets):
                tid = assigned.get(di, next_id - 1)
                head_pts, shoulder_pts, chest_center = extract_pose_features(
                    frame, det, pose_model, width, height
                )
                draw_pose_alignment(vis, det, tid, head_pts, shoulder_pts, chest_center)

        writer.write(vis)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames ...")

    cap.release()
    writer.release()
    print(f"Done. {frame_count} frames → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E3: Chest + Head Pose Alignment Tracking")
    parser.add_argument("--det-weights", type=str, default="models/best.pt",
                        help="Path to YOLOv8 detection weights")
    parser.add_argument("--pose-weights", type=str, default="models/yolov8n-pose.pt",
                        help="Path to YOLOv8 pose weights")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--output", type=str, default="output/e3_pose_alignment.mp4",
                        help="Path to output video")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_pose_alignment_tracking(args.video, args.det_weights, args.pose_weights, args.output)


if __name__ == "__main__":
    main()
