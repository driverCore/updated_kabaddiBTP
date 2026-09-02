"""
bytetrack_gt_association.py
===========================
E0 variant — ByteTrack on Ground-Truth Annotations.

Reads YOLO-format ground-truth annotations and feeds them (conf = 1.0)
into the ultralytics BYTETracker to assign consistent track IDs across
frames without any learned detector.

This demonstrates the ByteTrack association algorithm in isolation,
independent of detection quality.

Usage
-----
    python bytetrack_gt_association.py

    Adjust BASE, FRAMES_DIR, ANNOT_DIR below to point to your data.

Inputs
------
- Extracted video frames (JPG/PNG)
- YOLO-format annotation .txt files (class cx cy w h, normalised)

Output
------
- Annotated frames with track IDs
- Per-frame JSON track records
- Combined tracks.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths  — adjust these to your dataset layout
# ---------------------------------------------------------------------------
BASE = Path("data")

FRAMES_DIR = BASE / "frames"
ANNOT_DIR  = BASE / "annotations"
OUTPUT_DIR      = Path("output") / "bytetrack_gt"
OUTPUT_JSON_DIR = OUTPUT_DIR / "json"
COMBINED_JSON   = OUTPUT_DIR / "tracks.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Class name map (from classes.txt, 0-based)
# ---------------------------------------------------------------------------
CLASSES_FILE = ANNOT_DIR / "classes.txt"
class_names: list[str] = []
if CLASSES_FILE.exists():
    class_names = [l.strip() for l in CLASSES_FILE.read_text().splitlines() if l.strip()]

def get_class_name(cls_id: int) -> str:
    if 0 <= cls_id < len(class_names):
        return class_names[cls_id]
    return str(cls_id)

# ---------------------------------------------------------------------------
# Colour palette (by track_id)
# ---------------------------------------------------------------------------
PALETTE = [
    (255,  87,  51),
    ( 51, 153, 255),
    (255, 204,  51),
    (102, 255, 102),
    (255, 102, 204),
    (  0, 204, 204),
    (178, 102, 255),
    (255, 153,   0),
    ( 51, 255, 204),
    (255,  51, 102),
]

def color_for(track_id: int) -> tuple[int, int, int]:
    return PALETTE[track_id % len(PALETTE)]

# ---------------------------------------------------------------------------
# Sort frame paths numerically
# ---------------------------------------------------------------------------
def sort_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 10**12

# ---------------------------------------------------------------------------
# Minimal mock object that satisfies BYTETracker.update()
#
# BYTETracker.update(results) reads:
#   results.conf   -> (N,)   float32  confidence scores
#   results.cls    -> (N,)   float32  class ids
#   results.xywh   -> (N,4)  float32  cx,cy,w,h absolute pixels
#   results.xyxy   -> (N,4)  float32  x1,y1,x2,y2
#   results[bool_mask] -> sliced MockResults
# ---------------------------------------------------------------------------
class MockResults:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls  = cls.astype(np.float32)

        if len(xyxy):
            x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w  = x2 - x1
            h  = y2 - y1
            self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask):
        """Support boolean-mask / integer-array indexing."""
        return MockResults(self.xyxy[mask], self.conf[mask], self.cls[mask])


def main():
    frame_paths = sorted(
        [p for p in FRAMES_DIR.iterdir()
         if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")],
        key=sort_key,
    )

    if not frame_paths:
        raise FileNotFoundError(f"No images found in {FRAMES_DIR}")

    print(f"Found {len(frame_paths)} frames in {FRAMES_DIR}")

    # ---------------------------------------------------------------------------
    # Initialise BYTETracker with default ultralytics bytetrack.yaml parameters
    # ---------------------------------------------------------------------------
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    # Mirror of the ultralytics default bytetrack.yaml
    byte_cfg = IterableSimpleNamespace(
        **{
            "tracker_type":      "bytetrack",
            "track_high_thresh": 0.5,
            "track_low_thresh":  0.1,
            "new_track_thresh":  0.6,
            "track_buffer":      30,
            "match_thresh":      0.8,
            "fuse_score":        True,
        }
    )

    tracker = BYTETracker(args=byte_cfg)

    # ---------------------------------------------------------------------------
    # Process each frame
    # ---------------------------------------------------------------------------
    processed   = 0
    skipped     = 0
    all_records: list[dict] = []

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"  [WARN] Could not read {frame_path.name}, skipping.")
            skipped += 1
            continue

        img_h, img_w = frame.shape[:2]

        # ---- Load ground-truth annotation ---------------------------------
        annot_path = ANNOT_DIR / (frame_path.stem + ".txt")

        xyxy_rows: list[list[float]] = []
        cls_rows:  list[float]       = []

        if annot_path.exists():
            for raw_line in annot_path.read_text().splitlines():
                parts = raw_line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx_n   = float(parts[1])
                cy_n   = float(parts[2])
                bw_n   = float(parts[3])
                bh_n   = float(parts[4])

                # Normalised YOLO -> absolute pixel coords
                x1 = (cx_n - bw_n / 2) * img_w
                y1 = (cy_n - bh_n / 2) * img_h
                x2 = (cx_n + bw_n / 2) * img_w
                y2 = (cy_n + bh_n / 2) * img_h

                xyxy_rows.append([x1, y1, x2, y2])
                cls_rows.append(float(cls_id))

        # ---- Build MockResults (GT => confidence 1.0) ---------------------
        if xyxy_rows:
            xyxy_np = np.array(xyxy_rows, dtype=np.float32)
            conf_np = np.ones(len(xyxy_rows), dtype=np.float32)
            cls_np  = np.array(cls_rows,  dtype=np.float32)
        else:
            xyxy_np = np.empty((0, 4), dtype=np.float32)
            conf_np = np.empty((0,),   dtype=np.float32)
            cls_np  = np.empty((0,),   dtype=np.float32)

        mock_results = MockResults(xyxy_np, conf_np, cls_np)

        # ---- Run ByteTrack ------------------------------------------------
        tracks_arr = tracker.update(mock_results, img=None)

        # ---- Draw results on frame ----------------------------------------
        annotated = frame.copy()

        if len(tracks_arr) > 0:
            for row in tracks_arr:
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                tid   = int(row[4])
                color = color_for(tid)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"ID {tid}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                ty = max(y1 - 8, th + 8)
                cv2.rectangle(
                    annotated,
                    (x1, ty - th - bl),
                    (x1 + tw + 6, ty + bl),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        # ---- Build per-frame track record ---------------------------------
        frame_record: dict = {
            "frame": frame_path.stem,
            "image_w": img_w,
            "image_h": img_h,
            "tracks": [],
        }

        if len(tracks_arr) > 0:
            for row in tracks_arr:
                x1r, y1r, x2r, y2r = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                tid     = int(row[4])
                t_conf  = float(row[5])
                t_cls   = int(row[6])

                cx_n = ((x1r + x2r) / 2) / img_w
                cy_n = ((y1r + y2r) / 2) / img_h
                bw_n = (x2r - x1r) / img_w
                bh_n = (y2r - y1r) / img_h

                frame_record["tracks"].append({
                    "track_id":   tid,
                    "class_id":   t_cls,
                    "class_name": get_class_name(t_cls),
                    "conf":       round(t_conf, 4),
                    "bbox_xyxy":  [round(x1r, 2), round(y1r, 2),
                                   round(x2r, 2), round(y2r, 2)],
                    "bbox_yolo":  [t_cls,
                                   round(cx_n, 6), round(cy_n, 6),
                                   round(bw_n, 6), round(bh_n, 6)],
                })

        all_records.append(frame_record)

        # ---- Save per-frame JSON sidecar ----------------------------------
        json_path = OUTPUT_JSON_DIR / (frame_path.stem + ".json")
        with open(json_path, "w") as jf:
            json.dump(frame_record, jf, indent=2)

        # ---- Save annotated frame -----------------------------------------
        out_path = OUTPUT_DIR / frame_path.name
        cv2.imwrite(str(out_path), annotated)

        processed += 1
        if processed % 20 == 0 or processed == len(frame_paths):
            print(f"  Processed {processed}/{len(frame_paths)} frames ...")

    # ---- Save combined JSON -----------------------------------------------
    with open(COMBINED_JSON, "w") as jf:
        json.dump(all_records, jf, indent=2)

    print(f"\nDone. {processed} frames saved to:\n  {OUTPUT_DIR}")
    print(f"Per-frame JSON sidecars : {OUTPUT_JSON_DIR}")
    print(f"Combined JSON           : {COMBINED_JSON}")
    if skipped:
        print(f"  ({skipped} frames skipped - could not be read)")


if __name__ == "__main__":
    main()
