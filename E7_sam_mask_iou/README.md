# E7 — SAM Mask IoU

Tracks players using pixel-level segmentation masks from SAM (Segment Anything Model) instead of rectangular bounding-box IoU.

## Input
- Video file
- Trained YOLOv8 detection weights (`best.pt`)
- MobileSAM checkpoint (`mobile_sam.pt`)

## Main Cue
- **SAM segmentation masks**: Per-player silhouette masks generated from bounding-box prompts
- **Mask IoU**: Pixel-level intersection-over-union between player masks across frames

## Association Purpose
Transitions from rectangular bounding-box association to pixel-level silhouette mask IoU. This provides more precise overlap estimation when players have non-rectangular poses (e.g., lunging, diving), reducing false associations from bounding-box overlap of nearby players.

## Files

### `sam_mask_tracker.py`
Main tracker using ultralytics SAM integration. Generates masks from bbox prompts and overlays them for visualisation.

```
python sam_mask_tracker.py --det-weights models/best.pt \
                           --sam-weights models/mobile_sam.pt \
                           --video input.mp4
```

### `sam_depth_pipeline.py`
Combined SAM + MiDaS pipeline that runs automatic mask generation (segment everything) alongside depth estimation, computing per-mask median and relative depth values.

```
python sam_depth_pipeline.py --video input.mp4
```

## Dependencies
- `ultralytics` (YOLOv8 + SAM)
- `mobile_sam` (for automatic mask generation in `sam_depth_pipeline.py`)
- `opencv-python`
- `numpy`
- `torch`

## Required Model Checkpoints
Place the following in `models/`:
- `models/best.pt` — YOLOv8 detection weights
- `models/mobile_sam.pt` — MobileSAM checkpoint

Download MobileSAM from: https://github.com/ChaoningZhang/MobileSAM

(Model weights are NOT included in this repository)
