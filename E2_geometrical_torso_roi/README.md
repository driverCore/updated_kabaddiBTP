# E2 — Geometrical Torso ROI

Tracks players using a geometric torso region-of-interest (ROI) for more stable association.

## Input
- Video file or extracted frames
- Trained YOLOv8 detection weights (`best.pt`)

## Main Cue
- **Torso ROI**: A vertical sub-region from 20% to 80% of each bounding-box height
- Torso-to-torso IoU is used instead of full bounding-box IoU

## Association Purpose
Reduces noise from limb motion and partial occlusion at the extremities of the bounding box. The torso region is more spatially stable across frames than the full bounding box, improving association accuracy during fast movements.

## Key Parameters
| Parameter | Value | Description |
|---|---|---|
| Torso vertical range | 20%–80% of box height | Excludes head and legs |
| IoU threshold | 0.15 | Minimum torso IoU for a valid match |
| Detection confidence | 0.30 | YOLOv8 confidence threshold |

## Usage
```
python torso_roi_tracker.py --weights models/best.pt --video input.mp4
```

## Dependencies
- `ultralytics` (YOLOv8)
- `opencv-python`
- `numpy`
