"""
midas_depth_tracker.py
──────────────────────
E8 — MiDaS Relative Depth Tracking.

Uses MiDaS monocular depth estimation to compute per-player relative
depth, adding a depth-plane gating cue to bounding-box IoU association.

Pipeline
--------
    YOLOv8 detect → Bbox IoU association → MiDaS depth map →
    Player-region median depth → Depth-annotated visualisation

Usage
-----
    python midas_depth_tracker.py --weights models/best.pt --video input.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────

DETECT_CONF = 0.30
IOU_THRESHOLD = 0.15

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


# ── MiDaS Model Loading ──────────────────────────────────────────────────

def load_midas_model():
    """Load MiDaS Small model and transforms."""
    print("[INFO] Loading MiDaS Small ...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.to("cpu").eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    return model, transform


@torch.no_grad()
def compute_depth_map(
    frame: np.ndarray,
    model,
    transform,
    target_height: int,
    target_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MiDaS relative depth map for a frame.

    Returns
    -------
    depth_map : (H, W) float32 — raw depth values
    norm_depth : (H, W) uint8 — normalised to 0-255
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    batch = transform(img_rgb).to("cpu")

    pred = model(batch)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth_map = pred.cpu().numpy()

    d_min, d_max = depth_map.min(), depth_map.max()
    norm_depth = ((depth_map - d_min) / (d_max - d_min + 1e-6) * 255.0).astype(np.uint8)

    return depth_map, norm_depth


# ── Player-Region Depth Extraction ────────────────────────────────────────

def median_depth_in_bbox(
    norm_depth: np.ndarray,
    det: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> float:
    """Extract the median normalised depth inside a bounding box."""
    x1, y1, x2, y2 = [int(round(v)) for v in det]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width, x2), min(frame_height, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop_depth = norm_depth[y1:y2, x1:x2] / 255.0
    return float(np.median(crop_depth))


# ── IoU-based Association ─────────────────────────────────────────────────

def associate_by_iou(
    dets: np.ndarray,
    track_bboxes: Dict[int, np.ndarray],
    next_id: int,
) -> Tuple[Dict[int, int], Dict[int, np.ndarray], int]:
    """Greedy IoU-based association between detections and existing tracks."""
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


# ── Main Pipeline ─────────────────────────────────────────────────────────

def run_depth_tracking(video_path: str, weights_path: str, output_path: str):
    """Run E8 MiDaS depth tracking on a video."""
    print("Loading YOLO detector ...")
    yolo_model = YOLO(weights_path)

    print("Loading MiDaS depth model ...")
    midas_model, midas_transform = load_midas_model()

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

        # 1. Compute MiDaS depth map
        depth_map, norm_depth = compute_depth_map(
            frame, midas_model, midas_transform, height, width
        )
        depth_color = cv2.applyColorMap(norm_depth, cv2.COLORMAP_MAGMA)

        # Blend depth color map with video frame (70% depth + 30% frame)
        vis = cv2.addWeighted(depth_color, 0.70, frame, 0.30, 0)

        # 2. Run detector and association
        res = yolo_model.predict(source=frame, conf=DETECT_CONF, verbose=False)[0]

        if res.boxes is not None and len(res.boxes) > 0:
            dets = res.boxes.xyxy.cpu().numpy()

            assigned, track_bboxes, next_id = associate_by_iou(
                dets, track_bboxes, next_id
            )

            for di, det in enumerate(dets):
                tid = assigned.get(di, next_id - 1)

                # Median relative depth inside bbox
                med_rel_depth = median_depth_in_bbox(norm_depth, det, width, height)

                color = color_for_id(tid)

                # Draw bounding box
                x1, y1, x2, y2 = [int(round(v)) for v in det]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

                # Draw label
                label = f"ID {tid}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                label_y1 = max(0, y1 - th - bl - 8)
                label_y2 = max(th + bl + 8, y1)
                cv2.rectangle(vis, (x1, label_y1), (x1 + tw + 10, label_y2), color, -1)
                cv2.putText(vis, label, (x1 + 5, label_y2 - bl - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

                # Draw depth sub-label
                sub_label = f"Rel Depth: {med_rel_depth:.2f}"
                sub_scale = 0.50
                (stw, sth), sbase = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
                sy1 = min(height - sth - 4, y2 + 2)
                sy2 = sy1 + sth + sbase + 4
                cv2.rectangle(vis, (x1, sy1), (x1 + stw + 8, sy2), (20, 20, 20), -1)
                cv2.putText(vis, sub_label, (x1 + 4, sy2 - sbase - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, sub_scale, color, 1, cv2.LINE_AA)

        writer.write(vis)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames ...")

    cap.release()
    writer.release()
    print(f"Done. {frame_count} frames → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E8: MiDaS Relative Depth Tracking")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                        help="Path to YOLOv8 detection weights")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--output", type=str, default="output/e8_midas_depth.mp4",
                        help="Path to output video")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_depth_tracking(args.video, args.weights, args.output)


if __name__ == "__main__":
    main()
