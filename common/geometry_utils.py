"""
geometry_utils.py
─────────────────
Shared bounding-box geometry utilities used across multiple
tracking heuristic experiments.

Functions
---------
- bbox_iou       : Intersection-over-Union for two axis-aligned boxes
- bbox_center    : Centre point of a bounding box
- bbox_diagonal  : Diagonal length of a bounding box
- stable_torso_bbox : Crop the torso region (20%–80% of box height)
- expand_bbox    : Expand a bbox by a margin ratio
- point_inside_bbox : Check if a point falls inside a bbox
"""

import numpy as np


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute intersection-over-union for two [x1, y1, x2, y2] boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Return the centre point [cx, cy] of a bounding box."""
    return np.array(
        [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=float
    )


def bbox_diagonal(bbox: np.ndarray) -> float:
    """Return the diagonal length of a bounding box."""
    w = max(1.0, float(bbox[2] - bbox[0]))
    h = max(1.0, float(bbox[3] - bbox[1]))
    return float(np.hypot(w, h))


def stable_torso_bbox(bbox: np.ndarray) -> np.ndarray:
    """
    Extract the geometric torso region from a full bounding box.

    Crops to the vertical band from 20% to 80% of the box height,
    which corresponds to the torso area (excluding head and legs).

    Parameters
    ----------
    bbox : array [x1, y1, x2, y2]

    Returns
    -------
    ndarray [x1, torso_y1, x2, torso_y2]
    """
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    h = y2 - y1
    return np.array([x1, y1 + 0.20 * h, x2, y1 + 0.80 * h], dtype=float)


def expand_bbox(bbox: np.ndarray, margin_ratio: float) -> np.ndarray:
    """Expand a bounding box by a fractional margin on each side."""
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    x_margin = width * margin_ratio
    y_margin = height * margin_ratio
    return np.array(
        [bbox[0] - x_margin, bbox[1] - y_margin,
         bbox[2] + x_margin, bbox[3] + y_margin],
        dtype=float,
    )


def point_inside_bbox(point: np.ndarray, bbox: np.ndarray) -> bool:
    """Check whether a 2-D point lies inside a bounding box."""
    return bool(
        bbox[0] <= point[0] <= bbox[2]
        and bbox[1] <= point[1] <= bbox[3]
    )
