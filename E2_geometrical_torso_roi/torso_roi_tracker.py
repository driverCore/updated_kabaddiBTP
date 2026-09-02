"""
torso_roi_tracker.py
────────────────────
E2 — Geometrical Torso ROI Tracking.

Tracks players by computing a stable torso ROI (20%–80% of bounding-box
height) for each detection and associating tracks based on torso-to-torso
IoU rather than full bounding-box IoU.

This reduces noise from limb motion and occlusion at the extremities
of the bounding box.

Pipeline
--------
    YOLOv8 detect → Extract torso ROI → Torso IoU matching → Track IDs

Usage
-----
    python torso_roi_tracker.py --weights models/best.pt --video input.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from ultralytics import YOLO

# ── Geometry Utilities (also available in common/geometry_utils.py) ────────

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


def stable_torso_bbox(bbox: np.ndarray) -> np.ndarray:
    """
    Extract the geometric torso region from a full bounding box.
    Crops to the vertical band from 20% to 80% of the box height.
    """
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    h = y2 - y1
    return np.array([x1, y1 + 0.20 * h, x2, y1 + 0.80 * h], dtype=float)


# ── Configuration ─────────────────────────────────────────────────────────

DETECT_CONF = 0.30
TORSO_IOU_THRESHOLD = 0.15   # minimum torso IoU to consider a match

# Colors (BGR)
PALETTE = [
    (255, 87, 51), (51, 153, 255), (255, 204, 51), (102, 255, 102),
    (255, 102, 204), (0, 204, 204), (230, 126, 34), (155, 89, 182),
    (46, 204, 113), (52, 152, 219), (241, 196, 15), (231, 76, 60),
]


def color_for_id(tid: int) -> tuple:
    return PALETTE[tid % len(PALETTE)]


# ── Torso ROI Association ─────────────────────────────────────────────────

def associate_torso_roi(
    dets: np.ndarray,
    prev_torso_tracks: Dict[int, np.ndarray],
    next_id: int,
) -> tuple[Dict[int, int], Dict[int, np.ndarray], int]:
    """
    Associate detections to existing tracks using torso ROI IoU.

    Parameters
    ----------
    dets : (N, 4) array of [x1, y1, x2, y2] detections
    prev_torso_tracks : dict mapping track_id → previous torso bbox
    next_id : next available track ID

    Returns
    -------
    assigned : dict mapping detection_index → track_id
    new_torso_tracks : dict mapping track_id → current torso bbox
    next_id : updated next available track ID
    """
    torso_dets = [stable_torso_bbox(d) for d in dets]

    assigned: Dict[int, int] = {}
    used_tracks: set = set()
    track_ids = list(prev_torso_tracks.keys())

    if track_ids:
        iou_mat = np.zeros((len(dets), len(track_ids)), dtype=float)
        for di, td in enumerate(torso_dets):
            for ti, tid in enumerate(track_ids):
                iou_mat[di, ti] = bbox_iou(td, prev_torso_tracks[tid])

        while True:
            if iou_mat.size == 0 or iou_mat.max() < TORSO_IOU_THRESHOLD:
                break
            bi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            di, ti = int(bi[0]), int(bi[1])
            if di not in assigned and ti not in used_tracks:
                assigned[di] = track_ids[ti]
                used_tracks.add(ti)
            iou_mat[di, :] = 0.0
            iou_mat[:, ti] = 0.0

    new_torso_tracks = {}
    for di, det in enumerate(dets):
        tid = assigned.get(di, next_id)
        if di not in assigned:
            next_id += 1
        new_torso_tracks[tid] = torso_dets[di]

    return assigned, new_torso_tracks, next_id


# ── Main Pipeline ─────────────────────────────────────────────────────────

def run_torso_roi_tracking(video_path: str, weights_path: str, output_path: str):
    """Run E2 torso ROI tracking on a video."""
    yolo_model = YOLO(weights_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_torso_tracks: Dict[int, np.ndarray] = {}
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
            assigned, prev_torso_tracks, next_id = associate_torso_roi(
                dets, prev_torso_tracks, next_id
            )

            torso_dets = [stable_torso_bbox(d) for d in dets]
            for di, det in enumerate(dets):
                tid = assigned.get(di, next_id - 1)
                color = color_for_id(tid)

                # Draw full bounding box
                x1, y1, x2, y2 = [int(round(v)) for v in det]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"ID {tid}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                label_y1 = max(0, y1 - th - bl - 8)
                label_y2 = max(th + bl + 8, y1)
                cv2.rectangle(vis, (x1, label_y1), (x1 + tw + 10, label_y2), color, -1)
                cv2.putText(vis, label, (x1 + 5, label_y2 - bl - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

                # Draw torso ROI (20%–80% height)
                td = torso_dets[di]
                tx1, ty1, tx2, ty2 = [int(round(v)) for v in td]
                cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)
                cv2.putText(vis, "Torso ROI", (tx1 + 3, ty1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        writer.write(vis)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames ...")

    cap.release()
    writer.release()
    print(f"Done. {frame_count} frames → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E2: Geometrical Torso ROI Tracking")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                        help="Path to YOLOv8 detection weights")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--output", type=str, default="output/e2_torso_roi.mp4",
                        help="Path to output video")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_torso_roi_tracking(args.video, args.weights, args.output)


if __name__ == "__main__":
    main()
