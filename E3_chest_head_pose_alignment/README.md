# E3 — Chest + Head Pose Alignment

Tracks players using upper-body pose keypoints (chest/shoulders + head/nose/eyes) for anatomical alignment-based association.

## Input
- Video file
- Trained YOLOv8 detection weights (`best.pt`)
- YOLOv8 pose estimation weights (`yolov8n-pose.pt`)

## Main Cue
- **Head keypoints**: Nose, left/right eyes, left/right ears (COCO indices 0–4)
- **Shoulder keypoints**: Left/right shoulders (COCO indices 5–6)
- **Chest centre**: Midpoint of left and right shoulders
- **Head–chest alignment axis**: Line connecting head centroid to chest centre

## Association Purpose
Provides an anatomical reference frame for each player. The chest centre and head centroid positions are more robust to limb-pose variation than full bounding-box geometry, helping maintain identity during rapid movements and close interactions.

## Key Parameters
| Parameter | Value | Description |
|---|---|---|
| Keypoint confidence | 0.30 | Minimum confidence to use a keypoint |
| IoU threshold | 0.15 | Minimum bbox IoU for track association |
| Detection confidence | 0.30 | YOLOv8 confidence threshold |

## Usage
```
python pose_alignment_tracker.py --det-weights models/best.pt \
                                 --pose-weights models/yolov8n-pose.pt \
                                 --video input.mp4
```

## Dependencies
- `ultralytics` (YOLOv8 detection + pose)
- `opencv-python`
- `numpy`

## Required Models
- Detection weights: `models/best.pt`
- Pose weights: `models/yolov8n-pose.pt`

(Not included in this repository)
