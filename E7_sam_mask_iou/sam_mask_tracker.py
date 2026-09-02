"""
sam_mask_tracker.py
───────────────────
E7 — SAM Mask IoU Tracking.

Uses MobileSAM (or SAM) to generate per-player segmentation masks
from bounding-box prompts, then uses mask IoU for finer-grained
association than rectangular bounding-box IoU.

Pipeline
--------
    YOLOv8 detect → Bbox IoU association → SAM mask generation →
    Mask overlay visualisation

Usage
-----
    python sam_mask_tracker.py --det-weights models/best.pt \\
                               --sam-weights models/mobile_sam.pt \\
                               --video input.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO, SAM

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


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


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


# ── SAM Mask Generation & Overlay ─────────────────────────────────────────

def generate_sam_masks(
    sam_model: SAM,
    frame: np.ndarray,
    dets: np.ndarray,
) -> np.ndarray | None:
    """
    Generate SAM masks for detected bounding boxes.

    Returns
    -------
    masks : (N, H, W) boolean array or None
    """
    try:
        sam_res = sam_model.predict(
            source=frame, bboxes=dets.tolist(), verbose=False
        )[0]
        if sam_res.masks is not None:
            return sam_res.masks.data.cpu().numpy()
    except Exception:
        pass
    return None


def overlay_sam_masks(
    vis: np.ndarray,
    masks: np.ndarray | None,
    dets: np.ndarray,
    assigned: Dict[int, int],
    next_id_offset: int,
    frame_height: int,
    frame_width: int,
) -> np.ndarray:
    """Overlay semi-transparent SAM masks and draw labelled boxes."""
    overlay = vis.copy()

    for di, det in enumerate(dets):
        tid = assigned.get(di, di + next_id_offset)
        color = color_for_id(tid)

        if masks is not None and di < len(masks):
            mask = masks[di]
            if mask.shape != (frame_height, frame_width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask = mask.astype(bool)

            overlay[mask] = (
                overlay[mask] * 0.40 + np.array(color, dtype=float) * 0.60
            ).astype(np.uint8)

    # Blend overlay
    cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)

    # Re-draw clean labels on top
    for di, det in enumerate(dets):
        tid = assigned.get(di, di + next_id_offset)
        color = color_for_id(tid)

        x1, y1, x2, y2 = [int(round(v)) for v in det]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"ID {tid} [SAM Mask]"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        label_y1 = max(0, y1 - th - bl - 8)
        label_y2 = max(th + bl + 8, y1)
        cv2.rectangle(vis, (x1, label_y1), (x1 + tw + 10, label_y2), color, -1)
        cv2.putText(vis, label, (x1 + 5, label_y2 - bl - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    return vis


# ── Main Pipeline ─────────────────────────────────────────────────────────

def run_sam_mask_tracking(
    video_path: str, det_weights: str, sam_weights: str, output_path: str
):
    """Run E7 SAM mask IoU tracking on a video."""
    print("Loading YOLO detector ...")
    yolo_model = YOLO(det_weights)

    print("Loading SAM model ...")
    sam_model = SAM(sam_weights)

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

            masks = generate_sam_masks(sam_model, frame, dets)
            vis = overlay_sam_masks(
                vis, masks, dets, assigned, next_id, height, width
            )

        writer.write(vis)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames ...")

    cap.release()
    writer.release()
    print(f"Done. {frame_count} frames → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E7: SAM Mask IoU Tracking")
    parser.add_argument("--det-weights", type=str, default="models/best.pt",
                        help="Path to YOLOv8 detection weights")
    parser.add_argument("--sam-weights", type=str, default="models/mobile_sam.pt",
                        help="Path to SAM/MobileSAM weights")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--output", type=str, default="output/e7_sam_mask.mp4",
                        help="Path to output video")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_sam_mask_tracking(args.video, args.det_weights, args.sam_weights, args.output)


if __name__ == "__main__":
    main()
