# E0 — Baseline YOLOv8 + ByteTrack

Baseline multi-object tracking using YOLOv8 detection with ByteTrack association.

## Input
- Video frames (extracted as images) or a video file
- Trained YOLOv8 detection weights (`best.pt`)

## Main Cue
- Bounding-box IoU overlap between consecutive frames
- ByteTrack's two-stage association (high-confidence then low-confidence detections)

## Association Purpose
Establishes the baseline tracking performance. All subsequent heuristics (E2–E8) build upon or modify this association strategy.

## Files

### `baseline_bytetrack.py`
CLI-driven ByteTrack on extracted frames using `model.track()` with the standard `bytetrack.yaml` configuration.

```
python baseline_bytetrack.py --weights models/best.pt --source frames/
```

### `bytetrack_gt_association.py`
Runs ByteTrack association on ground-truth annotations (confidence = 1.0) to evaluate the tracker in isolation from detection quality. Uses `BYTETracker` directly with `MockResults`.

## Dependencies
- `ultralytics` (YOLOv8)
- `opencv-python`
- `numpy`

## Required Model
Place the trained YOLOv8 detection weights at `models/best.pt` (not included in this repository).
