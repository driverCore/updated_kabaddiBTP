# Kabaddi BTP — Player Tracking & Action Recognition

Source code for kabaddi player tracking heuristics and skeleton-based action recognition.

## Project Structure

```
Kabaddi Player Tracking & Action Recognition
│
├── Tracking Heuristics
│   ├── E0 — Baseline YOLOv8 + ByteTrack
│   ├── E2 — Geometrical Torso ROI
│   ├── E3 — Chest + Head Pose Alignment
│   ├── E5 — Hip/Shorts Colour Sampling
│   ├── E7 — SAM Mask IoU
│   └── E8 — MiDaS Relative Depth
│
└── Skeleton-Based Action Recognition
    ├── YOLOv8 Detection → YOLOv8-Pose (17 COCO Keypoints)
    ├── Skeleton Sequence Processing
    └── ST-GCN / MMAction2 (12-class Kabaddi Actions)
```

## Tracking Heuristics

| ID | Heuristic | Description |
|----|-----------|-------------|
| E0 | Baseline YOLOv8 + ByteTrack | Standard multi-object tracking baseline |
| E2 | Geometrical Torso ROI | Torso region (20%–80% height) IoU association |
| E3 | Chest + Head Pose Alignment | Upper-body keypoint alignment cue |
| E5 | Hip/Shorts Colour Sampling | Team-aware identity via HSV colour classification |
| E7 | SAM Mask IoU | Pixel-level silhouette mask association |
| E8 | MiDaS Relative Depth | Monocular depth-plane gating cue |

See each experiment folder for detailed documentation.

## Action Recognition

12-class skeleton-based action recognition using ST-GCN on COCO 17-keypoint skeletons extracted via YOLOv8-Pose.

| ID | Action | ID | Action |
|----|--------|----|--------|
| 0 | ankle_hold | 6 | hand_touch |
| 1 | back_kick | 7 | lion_jump |
| 2 | block | 8 | side_kick |
| 3 | chain_tackle | 9 | thigh_hold |
| 4 | dash | 10 | toe_touch |
| 5 | dubki | 11 | waist_hold |

See `action_recognition/README.md` for setup, training, and inference instructions.

## Repository Layout

```
├── E0_baseline_bytetrack/          # Baseline YOLOv8 + ByteTrack
├── E2_geometrical_torso_roi/       # Torso ROI slicing
├── E3_chest_head_pose_alignment/   # Pose keypoint alignment
├── E5_hip_shorts_colour/           # Hip/shorts colour sampling
├── E7_sam_mask_iou/                # SAM mask IoU
├── E8_midas_depth/                 # MiDaS relative depth
├── common/                         # Shared tracking utilities
├── models/                         # Trained YOLOv8 detection model
├── action_recognition/             # ST-GCN action recognition
│   ├── configs/                    # MMAction2 configs (base + kabaddi)
│   ├── tools/                      # Patched train.py / test.py
│   ├── scripts/                    # Custom BTP scripts
│   ├── class_mapping.json          # 12-class action mapping
│   ├── run_experiment.py           # End-to-end experiment orchestrator
│   ├── requirements.txt            # Verified environment dependencies
│   └── README.md
├── .gitignore
├── README.md
└── requirements.txt                # Tracking heuristics dependencies
```

## Requirements

**Tracking heuristics:**
```bash
pip install -r requirements.txt
```

**Action recognition:**
```bash
pip install -r action_recognition/requirements.txt
```

See `action_recognition/README.md` for detailed setup instructions (PyTorch + OpenMMLab stack).

## Model Weights

- `models/best.pt` — Trained YOLOv8 kabaddi player detection model (included)
- `yolov8n-pose.pt` — YOLOv8 pose estimation weights (download from Ultralytics)
- `mobile_sam.pt` — MobileSAM checkpoint (download separately)
- MiDaS Small — Downloaded automatically via `torch.hub`
- ST-GCN checkpoint — Not included; train using `action_recognition/run_experiment.py`
