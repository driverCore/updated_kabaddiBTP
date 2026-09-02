# Kabaddi Tracking Heuristics

Source code for six player tracking heuristic experiments developed for the Kabaddi BTP project.

## Experiments

| ID | Heuristic | Description |
|----|-----------|-------------|
| E0 | Baseline YOLOv8 + ByteTrack | Standard multi-object tracking baseline |
| E2 | Geometrical Torso ROI | Torso region (20%–80% height) IoU association |
| E3 | Chest + Head Pose Alignment | Upper-body keypoint alignment cue |
| E5 | Hip/Shorts Colour Sampling | Team-aware identity via HSV colour classification |
| E7 | SAM Mask IoU | Pixel-level silhouette mask association |
| E8 | MiDaS Relative Depth | Monocular depth-plane gating cue |

## Repository Structure

```
tracking_heuristics/
│
├── E0_baseline_bytetrack/          # Baseline YOLOv8 + ByteTrack
│   ├── baseline_bytetrack.py       # CLI-driven ByteTrack on frames
│   ├── bytetrack_gt_association.py # ByteTrack on ground-truth annotations
│   └── README.md
│
├── E2_geometrical_torso_roi/       # Torso ROI slicing
│   ├── torso_roi_tracker.py        # Torso IoU association
│   └── README.md
│
├── E3_chest_head_pose_alignment/   # Pose keypoint alignment
│   ├── pose_alignment_tracker.py   # Head + shoulder alignment
│   └── README.md
│
├── E5_hip_shorts_colour/           # Hip/shorts colour sampling
│   ├── hip_colour_tracker.py       # Full TeamAwareIdentityTracker
│   ├── simple_team_color.py        # Lightweight HSV classifier
│   └── README.md
│
├── E7_sam_mask_iou/                # SAM mask IoU
│   ├── sam_mask_tracker.py         # SAM mask generation + overlay
│   ├── sam_depth_pipeline.py       # Combined SAM + MiDaS pipeline
│   └── README.md
│
├── E8_midas_depth/                 # MiDaS relative depth
│   ├── midas_depth_tracker.py      # Depth-annotated tracking
│   ├── depth_features.py           # Standalone depth extraction
│   └── README.md
│
├── common/                         # Shared utilities
│   ├── geometry_utils.py           # Bbox IoU, centre, diagonal, torso ROI
│   ├── kalman4d.py                 # 4-state Kalman filter
│   ├── hungarian_match.py          # Hungarian algorithm association
│   └── README.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Weights

Model weights are **not included** in this repository. Place them in a `models/` directory:

- `models/best.pt` — Trained YOLOv8 detection weights
- `models/yolov8n-pose.pt` — YOLOv8 pose estimation weights
- `models/mobile_sam.pt` — MobileSAM checkpoint

MiDaS Small is downloaded automatically via `torch.hub` on first run.
