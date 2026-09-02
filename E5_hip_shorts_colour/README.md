# E5 — Hip/Shorts Colour Sampling

Team-aware identity tracking using hip-joint/shorts colour as the primary appearance cue.

## Input
- Video file
- YOLOv8 pose estimation weights (`yolov8n-pose.pt`)
- YOLOv8 player detection weights (`best.pt`) for filtering

## Main Cue
- **Hip keypoint crop**: Uses COCO hip keypoints (indices 11, 12) to extract the shorts/pant region
- **Lower-body fallback**: When hip keypoints are unavailable, crops the bottom 55% of the bounding box
- **HSV colour classification**: White vs Blue pixel ratio thresholds to determine team (India=Blue, Iran=White)

## Association Purpose
Prevents inter-team ID switches by locking each display ID to one team colour. The `TeamAwareIdentityTracker` combines ByteTrack raw IDs (as hints only), predicted spatial position, and colour evidence to maintain stable per-team identity.

## Key Parameters
| Parameter | Value | Description |
|---|---|---|
| White threshold | 0.05 | Minimum white pixel ratio to classify as Iran |
| Blue threshold | 0.03 | Minimum blue pixel ratio to classify as India |
| White confidence margin | 0.02 | Extra margin for strong team evidence |
| Max missing frames | 45 | Frames before a track is purged |
| Centre gate factor | 0.8 | Spatial gate relative to bbox diagonal |
| Motion gate per frame | 8.0 | Additional gate per missing frame |
| Raw ID match bonus | 0.35 | Score bonus when ByteTrack ID matches |

## Files

### `hip_colour_tracker.py`
Full implementation with `TeamAwareIdentityTracker`, hip-joint pose crop, lower-body fallback, HSV classification, and team-aware spatial continuity.

### `simple_team_color.py`
Lightweight alternative using mean hue from the upper-body crop for quick jersey colour classification.

## Usage
```
python hip_colour_tracker.py --video input.mp4 \
                             --pose-model models/yolov8n-pose.pt \
                             --filter-model models/best.pt
```

## Dependencies
- `ultralytics` (YOLOv8 pose + detection)
- `opencv-python`
- `numpy`

## Required Models
- Pose weights: `models/yolov8n-pose.pt`
- Detection weights: `models/best.pt`

(Not included in this repository)
