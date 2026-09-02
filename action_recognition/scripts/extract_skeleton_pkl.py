"""
extract_skeleton_pkl.py
=======================
All-in-one pipeline for kabaddi skeleton extraction.

For every video in Updated_dataset/{class}/:
  1. best.pt       → detect kabaddi player bounding boxes (filters spectators)
  2. Crop + yolov8n-pose → 17-keypoint COCO skeleton per detected player
  3. Map keypoints back to original frame coordinates
  4. Save outputs:
       a) Annotated JPG  →  Updated_dataset/{cls}/{cls}_skeletal/frames_jpg/{video_name}/frame_XXXXXX.jpg
       b) JSON coords    →  Updated_dataset/{cls}/{cls}_skeletal/coordinates/{video_name}/frame_XXXXXX.json
       c) Aggregate into (M,T,V,2) arrays → kabaddi_train.pkl / kabaddi_val.pkl for STGCN

Coordinate JSON format (per frame):
  {
    "frame_idx": int,
    "img_shape": [H, W],
    "players": [
      {
        "player_id": int,
        "detect_conf": float,
        "bbox_xyxy": [x1, y1, x2, y2],
        "keypoints": [          <- 17 COCO keypoints
          {"id": int, "name": str, "x": float, "y": float, "conf": float},
          ...
        ]
      },
      ...
    ]
  }
"""

import os
import sys
import json
import random
import pickle
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not found.  pip install ultralytics")
    sys.exit(1)

# ─────────────────────────── CONFIGURATION ────────────────────────────────────
DATASET_ROOT = r"C:\Users\aryan\Downloads\BTP\Updated_dataset"
PKL_OUT_DIR  = r"C:\Users\aryan\Downloads\BTP\mmaction2\data\kabaddi"
DETECTOR_PT  = r"C:\Users\aryan\Downloads\BTP\best.pt"
POSE_PT      = r"C:\Users\aryan\Downloads\BTP\yolov8n-pose.pt"

MAX_PERSONS  = 2      # top-N players per frame stored in skeleton arrays
MAX_FRAMES   = 150    # uniform sub-sample cap per video
DETECT_CONF  = 0.30   # best.pt detection confidence
POSE_CONF    = 0.25   # yolov8n-pose confidence
CROP_MARGIN  = 0.10   # expand player box by 10% before pose crop
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42
NUM_KP       = 17
# ──────────────────────────────────────────────────────────────────────────────

# COCO 17-keypoint names
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO skeleton limb pairs (0-indexed)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # face
    (5, 6),                                     # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12),                          # torso sides
    (11, 12),                                  # hips
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

# Per-player colours (BGR) for visualization
PLAYER_COLORS = [
    (0, 255, 100),    # player 0 → green
    (255, 100, 0),    # player 1 → blue
    (0, 180, 255),    # player 2 → orange
    (200, 0, 255),    # player 3 → purple
]

random.seed(RANDOM_SEED)
os.makedirs(PKL_OUT_DIR, exist_ok=True)

# ── Load models ────────────────────────────────────────────────────────────────
print(f"Loading detector : {DETECTOR_PT}")
detector = YOLO(DETECTOR_PT)

print(f"Loading pose model: {POSE_PT}")
pose_model = YOLO(POSE_PT)

# ── Discover classes ───────────────────────────────────────────────────────────
classes = sorted([
    d for d in os.listdir(DATASET_ROOT)
    if os.path.isdir(os.path.join(DATASET_ROOT, d))
    and not d.endswith("_skeletal")   # skip any existing output folders
])
class_to_idx = {cls: i for i, cls in enumerate(classes)}

print(f"\nFound {len(classes)} classes:")
for i, c in enumerate(classes):
    print(f"  {i:2d}  {c}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def expand_box(x1, y1, x2, y2, margin, H, W):
    pw = (x2 - x1) * margin
    ph = (y2 - y1) * margin
    return (
        clamp(int(x1 - pw), 0, W - 1),
        clamp(int(y1 - ph), 0, H - 1),
        clamp(int(x2 + pw), 1, W),
        clamp(int(y2 + ph), 1, H),
    )


def draw_skeleton(frame, kps_xy, kps_conf, color, kp_thr=0.2):
    """Draw skeleton on frame in-place. kps_xy: (17,2), kps_conf: (17,)"""
    # Limbs
    for a, b in SKELETON:
        if kps_conf[a] >= kp_thr and kps_conf[b] >= kp_thr:
            pt1 = (int(kps_xy[a, 0]), int(kps_xy[a, 1]))
            pt2 = (int(kps_xy[b, 0]), int(kps_xy[b, 1]))
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
    # Joints
    for v in range(NUM_KP):
        if kps_conf[v] >= kp_thr:
            cx, cy = int(kps_xy[v, 0]), int(kps_xy[v, 1])
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, color, 1, cv2.LINE_AA)


def process_frame(frame, H, W):
    """
    Run best.pt + yolov8n-pose on one frame.

    Returns list of player dicts, each with:
        bbox      : [x1,y1,x2,y2]
        conf      : float
        kp_xy     : np.ndarray (17,2) in original frame coords
        kp_conf   : np.ndarray (17,)
    Sorted by detection confidence descending.
    """
    # ── Step 1: detect players ─────────────────────────────────────────────────
    det_res = detector(frame, conf=DETECT_CONF, verbose=False)
    raw_boxes = []
    if det_res and det_res[0].boxes is not None:
        for box in det_res[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            conf = float(box.conf[0].cpu().numpy())
            raw_boxes.append((x1, y1, x2, y2, conf))
    raw_boxes.sort(key=lambda b: -b[4])

    players = []
    for (bx1, by1, bx2, by2, bconf) in raw_boxes:
        # ── Step 2: crop with margin ───────────────────────────────────────────
        cx1, cy1, cx2, cy2 = expand_box(bx1, by1, bx2, by2, CROP_MARGIN, H, W)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        # ── Step 3: pose on crop ───────────────────────────────────────────────
        pose_res = pose_model(crop, conf=POSE_CONF, verbose=False)
        if not pose_res or pose_res[0].keypoints is None:
            continue

        kp_data = pose_res[0].keypoints
        xy   = kp_data.xy.cpu().numpy()    # (N, 17, 2) in crop coords
        conf = kp_data.conf.cpu().numpy()  # (N, 17)
        if len(xy) == 0:
            continue

        # Take highest-confidence pose in the crop
        best_pose = np.argmax(conf.sum(axis=1))
        kp_xy   = xy[best_pose].copy()      # (17, 2)
        kp_conf = conf[best_pose].copy()    # (17,)

        # ── Step 4: map back to original frame ─────────────────────────────────
        kp_xy[:, 0] += cx1
        kp_xy[:, 1] += cy1

        players.append(dict(
            bbox    = [float(bx1), float(by1), float(bx2), float(by2)],
            conf    = float(bconf),
            kp_xy   = kp_xy.astype(np.float32),
            kp_conf = kp_conf.astype(np.float32),
        ))

    return players


# ── Main extraction loop ───────────────────────────────────────────────────────
train_samples = []
val_samples   = []
VIDEO_EXTS    = (".mp4", ".avi", ".mov", ".mkv")

for cls in classes:
    cls_dir    = os.path.join(DATASET_ROOT, cls)
    label      = class_to_idx[cls]

    # Output subfolders inside the class dir
    skeletal_dir  = os.path.join(cls_dir, f"{cls}_skeletal")
    jpg_base_dir  = os.path.join(skeletal_dir, "frames_jpg")
    json_base_dir = os.path.join(skeletal_dir, "coordinates")
    os.makedirs(jpg_base_dir,  exist_ok=True)
    os.makedirs(json_base_dir, exist_ok=True)

    videos = sorted([
        f for f in os.listdir(cls_dir)
        if f.lower().endswith(VIDEO_EXTS)
    ])
    random.shuffle(videos)

    if len(videos) == 1:
        train_vids, val_vids = videos, []
    else:
        n_val      = max(1, int(len(videos) * VAL_SPLIT))
        val_vids   = videos[:n_val]
        train_vids = videos[n_val:]

    print(f"\n[{cls}]  train={len(train_vids)}  val={len(val_vids)}")

    for split_name, vid_list, sample_list in [
        ("train", train_vids, train_samples),
        ("val",   val_vids,   val_samples),
    ]:
        for vid_file in vid_list:
            vid_path   = os.path.join(cls_dir, vid_file)
            vid_stem   = os.path.splitext(vid_file)[0]   # e.g. "dash1"
            frame_dir  = f"{cls}/{vid_stem}"

            # Per-video output directories
            jpg_vid_dir  = os.path.join(jpg_base_dir,  vid_stem)
            json_vid_dir = os.path.join(json_base_dir, vid_stem)
            os.makedirs(jpg_vid_dir,  exist_ok=True)
            os.makedirs(json_vid_dir, exist_ok=True)

            print(f"  [{split_name}] {vid_file} ...", end="", flush=True)

            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print(" SKIPPED (cannot open)")
                continue

            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Uniform sub-sample
            if n_total <= MAX_FRAMES:
                frame_indices = list(range(n_total))
            else:
                step = n_total / MAX_FRAMES
                frame_indices = sorted(set(int(i * step) for i in range(MAX_FRAMES)))

            T          = len(frame_indices)
            frame_set  = set(frame_indices)
            frame_map  = {f: t for t, f in enumerate(frame_indices)}

            # PKL arrays
            kp_arr  = np.zeros((MAX_PERSONS, T, NUM_KP, 2), dtype=np.float32)
            kps_arr = np.zeros((MAX_PERSONS, T, NUM_KP),    dtype=np.float32)

            cur = 0
            ret = True
            while ret and cur <= max(frame_indices):
                ret, frame = cap.read()
                if not ret:
                    break

                if cur in frame_set:
                    t       = frame_map[cur]
                    players = process_frame(frame, H, W)
                    top_p   = players[:MAX_PERSONS]

                    # ── Fill PKL arrays ────────────────────────────────────────
                    for p_idx, pl in enumerate(top_p):
                        kp_arr[p_idx, t]  = pl["kp_xy"]
                        kps_arr[p_idx, t] = pl["kp_conf"]

                    # ── Draw skeleton on a copy of the frame ───────────────────
                    vis = frame.copy()
                    for p_idx, pl in enumerate(players):   # draw ALL detected players
                        color = PLAYER_COLORS[p_idx % len(PLAYER_COLORS)]
                        bx1, by1, bx2, by2 = [int(v) for v in pl["bbox"]]

                        # Bounding box from best.pt
                        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
                        cv2.putText(
                            vis,
                            f"P{p_idx} {pl['conf']:.2f}",
                            (bx1, max(15, by1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                        )

                        # Skeleton keypoints
                        draw_skeleton(vis, pl["kp_xy"], pl["kp_conf"], color)

                    # ── Save JPG ───────────────────────────────────────────────
                    jpg_path = os.path.join(jpg_vid_dir, f"frame_{cur:06d}.jpg")
                    cv2.imwrite(jpg_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    # ── Save JSON coordinates ──────────────────────────────────
                    frame_data = {
                        "frame_idx": cur,
                        "img_shape": [H, W],
                        "players": []
                    }
                    for p_idx, pl in enumerate(players):
                        kp_list = []
                        for v in range(NUM_KP):
                            kp_list.append({
                                "id":   v,
                                "name": KP_NAMES[v],
                                "x":    round(float(pl["kp_xy"][v, 0]), 2),
                                "y":    round(float(pl["kp_xy"][v, 1]), 2),
                                "conf": round(float(pl["kp_conf"][v]),  4),
                            })
                        frame_data["players"].append({
                            "player_id":   p_idx,
                            "detect_conf": round(pl["conf"], 4),
                            "bbox_xyxy":   [round(v, 1) for v in pl["bbox"]],
                            "keypoints":   kp_list,
                        })

                    json_path = os.path.join(json_vid_dir, f"frame_{cur:06d}.json")
                    with open(json_path, "w") as jf:
                        json.dump(frame_data, jf, indent=2)

                cur += 1

            cap.release()

            # Build PKL sample dict for this video
            sample = dict(
                frame_dir      = frame_dir,
                label          = label,
                img_shape      = (H, W),
                original_shape = (H, W),
                total_frames   = T,
                keypoint       = kp_arr,
                keypoint_score = kps_arr,
            )
            sample_list.append(sample)

            # Count actual saved files for progress
            n_jpg  = len(os.listdir(jpg_vid_dir))
            n_json = len(os.listdir(json_vid_dir))
            print(f" done  ({T} frames | {n_jpg} jpgs | {n_json} jsons)")


# ── Save PKL files ─────────────────────────────────────────────────────────────
train_pkl    = os.path.join(PKL_OUT_DIR, "kabaddi_train.pkl")
val_pkl      = os.path.join(PKL_OUT_DIR, "kabaddi_val.pkl")
combined_pkl = os.path.join(PKL_OUT_DIR, "kabaddi_skeleton.pkl")

with open(train_pkl, "wb") as f:
    pickle.dump(train_samples, f)
with open(val_pkl, "wb") as f:
    pickle.dump(val_samples, f)

combined = dict(
    split=dict(
        train=[s["frame_dir"] for s in train_samples],
        val  =[s["frame_dir"] for s in val_samples],
    ),
    annotations=train_samples + val_samples,
)
with open(combined_pkl, "wb") as f:
    pickle.dump(combined, f)

# classes.txt
with open(os.path.join(PKL_OUT_DIR, "classes.txt"), "w") as f:
    for c in classes:
        f.write(c + "\n")

print("\n" + "=" * 65)
print(f"Train pkl  : {train_pkl}  ({len(train_samples)} samples)")
print(f"Val   pkl  : {val_pkl}    ({len(val_samples)} samples)")
print(f"Combined   : {combined_pkl}")
print(f"\nClasses ({len(classes)}):")
for i, c in enumerate(classes):
    print(f"  {i:2d}  {c}")
print("=" * 65)
print("\nOutput folder structure per class:")
print("  Updated_dataset/{cls}/{cls}_skeletal/")
print("    frames_jpg/{video_name}/frame_XXXXXX.jpg   <- skeleton visualizations")
print("    coordinates/{video_name}/frame_XXXXXX.json <- keypoint coordinates")
print("\nNext — train STGCN:")
print("  cd mmaction2")
print("  python tools/train.py configs/skeleton/stgcn/kabaddi_stgcn.py")
