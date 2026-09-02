# E8 — MiDaS Relative Depth

Tracks players using monocular relative depth estimation from MiDaS as an additional cue for depth-plane gating.

## Input
- Video file or extracted frames
- Trained YOLOv8 detection weights (`best.pt`)
- MiDaS model (downloaded automatically via `torch.hub`)

## Main Cue
- **MiDaS depth map**: Per-frame monocular relative depth estimated by MiDaS Small
- **Player-region median depth**: Median depth value inside each player's bounding box
- **Depth-colour overlay**: Magma colormap visualisation of the depth field

## Association Purpose
Adds a relative-depth plane dimension to the association. Players at different depths (foreground raiders vs background defenders) can be distinguished even when their bounding boxes overlap in the 2D image plane.

## Key Parameters
| Parameter | Value | Description |
|---|---|---|
| MiDaS model | MiDaS_small | Loaded via torch.hub |
| Depth normalisation | min-max to [0, 255] | Per-frame normalisation |
| Depth overlay blend | 70% depth + 30% frame | Visualisation blend ratio |
| IoU threshold | 0.15 | Minimum bbox IoU for association |
| Detection confidence | 0.30 | YOLOv8 confidence threshold |

## Files

### `midas_depth_tracker.py`
Main tracker computing per-frame depth maps and annotating players with relative depth values.

```
python midas_depth_tracker.py --weights models/best.pt --video input.mp4
```

### `depth_features.py`
Standalone depth feature extraction. Loads track records and computes per-player median depth for use with the Hungarian matcher.

## Dependencies
- `ultralytics` (YOLOv8)
- `opencv-python`
- `numpy`
- `torch`
- `timm` (dependency of MiDaS, installed automatically)

## Model Download
MiDaS Small is downloaded automatically via `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")` on first run. No manual checkpoint download is required.
