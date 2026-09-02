# Common Utilities

Shared modules used by multiple tracking heuristic experiments.

## Modules

### `geometry_utils.py`
Bounding-box geometry functions:
- `bbox_iou` — Intersection-over-Union between two `[x1, y1, x2, y2]` boxes
- `bbox_center` — Centre point of a bounding box
- `bbox_diagonal` — Diagonal length of a bounding box
- `stable_torso_bbox` — Extract the torso region (20%–80% of box height)
- `expand_bbox` — Expand a bbox by a margin ratio
- `point_inside_bbox` — Check if a point falls inside a bbox

### `kalman4d.py`
A lightweight 4-state constant-velocity Kalman filter for tracking bounding-box centres.

- State vector: `[cx, cy, vx, vy]`
- Measurement: `[cx, cy]`
- Uses piecewise-constant white-noise acceleration model for process noise
- Joseph-form covariance update for numerical stability

### `hungarian_match.py`
Frame-by-frame player association using the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`).

- Cost = `(1 − IoU) + λ · |depth_track − depth_det|`
- IoU-based gating to prevent impossible matches
- Integrates depth cues when available

#### Dependencies
- `numpy`
- `scipy`
