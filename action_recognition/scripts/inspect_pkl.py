"""
Inspect existing kabaddi PKL files and verify they are STGCN-ready.
Also checks for corrupted/empty samples and prints a full report.
"""
import pickle
import numpy as np
import sys

PKL_TRAIN = r"C:\Users\aryan\Downloads\BTP\mmaction2\data\kabaddi\kabaddi_train.pkl"
PKL_VAL   = r"C:\Users\aryan\Downloads\BTP\mmaction2\data\kabaddi\kabaddi_val.pkl"

REQUIRED_KEYS = {"frame_dir", "label", "keypoint", "keypoint_score", "total_frames", "img_shape"}
NUM_KP = 17

errors   = []
warnings = []

for pkl_path, split_name in [(PKL_TRAIN, "TRAIN"), (PKL_VAL, "VAL")]:
    print(f"\n{'='*60}")
    print(f"  {split_name}: {pkl_path}")
    print(f"{'='*60}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"  Type     : {type(data)}")
    print(f"  Samples  : {len(data)}")

    labels = set()
    class_counts = {}
    empty_kp_count = 0

    for i, sample in enumerate(data):
        sid = sample.get("frame_dir", f"sample_{i}")

        # Check required keys
        missing = REQUIRED_KEYS - set(sample.keys())
        if missing:
            errors.append(f"[{split_name}][{sid}] Missing keys: {missing}")
            continue

        kp  = sample["keypoint"]        # should be (M, T, V, 2)
        kps = sample["keypoint_score"]  # should be (M, T, V)
        lbl = sample["label"]
        T   = sample["total_frames"]

        labels.add(lbl)
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

        # Shape checks
        if kp.ndim != 4:
            errors.append(f"[{split_name}][{sid}] keypoint ndim={kp.ndim}, expected 4")
        elif kp.shape[2] != NUM_KP:
            errors.append(f"[{split_name}][{sid}] keypoint joints={kp.shape[2]}, expected {NUM_KP}")
        elif kp.shape[3] != 2:
            errors.append(f"[{split_name}][{sid}] keypoint last_dim={kp.shape[3]}, expected 2")

        # Check T matches actual array dim
        if kp.ndim == 4 and kp.shape[1] != T:
            warnings.append(f"[{split_name}][{sid}] total_frames={T} but kp.shape[1]={kp.shape[1]}")

        # Check for all-zero keypoints (no detection in any frame)
        if kp.ndim == 4:
            nonzero_frames = np.any(kp != 0, axis=(0, 2, 3)).sum()
            if nonzero_frames == 0:
                empty_kp_count += 1
                warnings.append(f"[{split_name}][{sid}] ALL keypoints are zero (no player detected in any frame)")
            elif nonzero_frames < T * 0.3:
                warnings.append(f"[{split_name}][{sid}] Sparse: only {nonzero_frames}/{T} frames have keypoints")

        # Label range
        if lbl < 0 or lbl > 11:
            errors.append(f"[{split_name}][{sid}] label={lbl} out of range [0,11]")

    print(f"\n  Labels found : {sorted(labels)}  ({len(labels)} unique)")
    print(f"  Samples/class:")
    for lbl in sorted(class_counts.keys()):
        print(f"    class {lbl:2d} : {class_counts[lbl]} video(s)")
    print(f"\n  Keypoint shape of first sample : {data[0]['keypoint'].shape}")
    print(f"  Frames with all-zero kp         : {empty_kp_count}/{len(data)}")

    if split_name == "TRAIN":
        kp0 = data[0]['keypoint']
        print(f"\n  Sample[0] frame_dir  : {data[0]['frame_dir']}")
        print(f"  Sample[0] label      : {data[0]['label']}")
        print(f"  Sample[0] total_frames: {data[0]['total_frames']}")
        print(f"  Sample[0] kp.shape   : {kp0.shape}  <- (M_persons, T_frames, V_joints, 2)")
        print(f"  Sample[0] kp.dtype   : {kp0.dtype}")
        print(f"  Sample[0] kp.max()   : {kp0.max():.2f}")
        print(f"  Sample[0] img_shape  : {data[0]['img_shape']}")

print(f"\n{'='*60}")
print("VERIFICATION SUMMARY")
print(f"{'='*60}")
if errors:
    print(f"\n  ERRORS ({len(errors)}):")
    for e in errors:
        print(f"    ✗ {e}")
else:
    print("  ✓ No errors found")

if warnings:
    print(f"\n  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"    ⚠ {w}")
else:
    print("  ✓ No warnings")

print(f"\n  PKL format : OK for MMAction2 PoseDataset")
print(f"  STGCN ready: {'YES' if not errors else 'NEEDS FIX'}")
