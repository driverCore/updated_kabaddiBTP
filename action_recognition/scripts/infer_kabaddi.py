"""
infer_kabaddi.py
================
Run skeleton-based inference on test samples from the validation PKL.

Generates:
  - Predicted class + confidence
  - Top-5 predictions per sample
  - Per-sample CSV results
  - Skeleton visualization (matplotlib) per sample

Usage:
  cd C:/Users/aryan/Downloads/BTP/mmaction2
  python infer_kabaddi.py work_dirs/stgcn_kabaddi/best_*.pth
  python infer_kabaddi.py                              (auto-detects checkpoint)
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import csv

# ─────────────── CLASS NAMES ──────────────────────────────────────────────────
CLASS_NAMES = [
    "ankle hold",    # 0
    "block",         # 1
    "chain tackle",  # 2
    "dash",          # 3
    "thigh hold",    # 4
    "waist hold",    # 5
    "back kick",     # 6
    "dubki",         # 7
    "hand touch",    # 8
    "lion jump",     # 9
    "side kick",     # 10
    "toe touch",     # 11
]

# COCO-17 skeleton connections for visualization
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # face
    (5, 6),                             # shoulders
    (5, 7), (7, 9),                    # left arm
    (6, 8), (8, 10),                   # right arm
    (5, 11), (6, 12),                  # torso sides
    (11, 12),                          # hips
    (11, 13), (13, 15),                # left leg
    (12, 14), (14, 16),                # right leg
]

WORK_DIR   = "work_dirs/stgcn_kabaddi"
CONFIG     = "configs/skeleton/stgcn/kabaddi_stgcn_v2.py"
VAL_PKL    = "data/kabaddi/kabaddi_val.pkl"
VIZ_DIR    = os.path.join(WORK_DIR, "inference_visualizations")
RESULTS_CSV = os.path.join(WORK_DIR, "inference_results.csv")


def find_best_checkpoint():
    if not os.path.exists(WORK_DIR):
        return None
    best = sorted([f for f in os.listdir(WORK_DIR) if "best_" in f and f.endswith(".pth")])
    if best:
        return os.path.join(WORK_DIR, best[-1])
    epochs = sorted([f for f in os.listdir(WORK_DIR) if f.startswith("epoch_") and f.endswith(".pth")])
    return os.path.join(WORK_DIR, epochs[-1]) if epochs else None


def run_inference(checkpoint_path):
    """Run mmaction2 inference on all validation samples."""
    import subprocess

    results_pkl = os.path.join(WORK_DIR, "infer_results.pkl")
    cmd = [
        sys.executable, "tools/test.py",
        CONFIG, checkpoint_path,
        "--dump", results_pkl,
    ]
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    # Load model outputs
    with open(results_pkl, "rb") as f:
        raw_results = pickle.load(f)

    # Load ground truth
    with open(VAL_PKL, "rb") as f:
        val_data = pickle.load(f)

    if isinstance(raw_results, list) and isinstance(raw_results[0], dict):
        scores_list = [r.get('pred_score', r.get('scores', [0]*12)) for r in raw_results]
    else:
        scores_list = raw_results

    scores_arr = np.array(scores_list)
    return val_data, scores_arr


def visualize_sample(sample, scores, sample_idx, gt_label, pred_label, top5):
    """Create a matplotlib visualization of skeleton + prediction."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        os.makedirs(VIZ_DIR, exist_ok=True)

        kp = sample['keypoint']      # (M, T, V, 2)
        kp_score = sample['keypoint_score']  # (M, T, V)
        M, T, V, _ = kp.shape

        # Pick middle frame
        mid_t = T // 2

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(
            f"Sample {sample_idx}: {CLASS_NAMES[gt_label]}\n"
            f"Predicted: {CLASS_NAMES[pred_label]} (conf={scores[pred_label]:.3f})",
            fontsize=13
        )

        colors = ['tab:green', 'tab:blue']

        # Left: skeleton at midframe
        ax = axes[0]
        ax.set_title("Skeleton at Middle Frame")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(1.2, -1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")

        for m in range(min(M, 2)):
            kp_t = kp[m, mid_t]          # (V, 2)
            sc_t = kp_score[m, mid_t]    # (V,)
            color = colors[m]
            # Draw skeleton lines
            for (a, b) in SKELETON_PAIRS:
                if sc_t[a] > 0.2 and sc_t[b] > 0.2:
                    ax.plot([kp_t[a, 0], kp_t[b, 0]],
                            [kp_t[a, 1], kp_t[b, 1]],
                            color=color, linewidth=2, alpha=0.8)
            # Draw joints
            valid = sc_t > 0.2
            ax.scatter(kp_t[valid, 0], kp_t[valid, 1],
                       c=color, s=40, zorder=5, label=f"Person {m}")

        patches = [mpatches.Patch(color=colors[i], label=f"Person {i}") for i in range(min(M, 2))]
        ax.legend(handles=patches, loc='upper right')

        # Right: bar chart of top-5 scores
        ax2 = axes[1]
        ax2.set_title("Top-5 Class Probabilities")
        top5_ids   = np.argsort(scores)[::-1][:5]
        top5_names = [CLASS_NAMES[i] for i in top5_ids]
        top5_vals  = scores[top5_ids]
        bar_colors = ['red' if i == pred_label else 'steelblue' for i in top5_ids]
        if gt_label in top5_ids:
            bar_colors[list(top5_ids).index(gt_label)] = 'gold'

        bars = ax2.barh(top5_names, top5_vals, color=bar_colors)
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("Confidence Score")
        for bar, val in zip(bars, top5_vals):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=10)

        legend_items = [
            mpatches.Patch(color='red',       label='Predicted'),
            mpatches.Patch(color='gold',      label='Ground Truth'),
            mpatches.Patch(color='steelblue', label='Other'),
        ]
        ax2.legend(handles=legend_items, loc='lower right')

        plt.tight_layout()
        out_path = os.path.join(VIZ_DIR, f"sample_{sample_idx:03d}_pred_{CLASS_NAMES[pred_label].replace(' ', '_')}.png")
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        return out_path
    except Exception as e:
        print(f"  Visualization error for sample {sample_idx}: {e}")
        return None


def main(checkpoint_path=None):
    print("=" * 70)
    print("PHASE 9/10 — INFERENCE ON VALIDATION SET")
    print("=" * 70)

    ckpt = checkpoint_path or find_best_checkpoint()
    if ckpt is None or not os.path.exists(ckpt):
        print("ERROR: No checkpoint found. Train first.")
        sys.exit(1)
    print(f"Checkpoint: {ckpt}")

    val_data, scores_arr = run_inference(ckpt)

    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    print(f"\nRunning inference on {len(val_data)} validation samples...")
    print(f"{'#':<6} {'Ground Truth':<22} {'Predicted':<22} {'Conf':>8} {'Correct':>8}")
    print("-" * 70)

    rows = []
    correct = 0
    for i, (sample, scores) in enumerate(zip(val_data, scores_arr)):
        gt_label   = sample['label']
        pred_label = int(np.argmax(scores))
        confidence = float(scores[pred_label])
        is_correct = gt_label == pred_label
        if is_correct:
            correct += 1

        top5_ids  = np.argsort(scores)[::-1][:5]
        top5_info = [(CLASS_NAMES[j], float(scores[j])) for j in top5_ids]

        flag = "OK" if is_correct else "WRONG"
        print(f"{i:<6} {CLASS_NAMES[gt_label]:<22} {CLASS_NAMES[pred_label]:<22} {confidence:>8.4f} {flag:>8}")
        print(f"       Top-5: {', '.join(f'{n}({s:.3f})' for n, s in top5_info)}")

        rows.append({
            "sample_idx":    i,
            "frame_dir":     sample.get("frame_dir", ""),
            "gt_label":      gt_label,
            "gt_class":      CLASS_NAMES[gt_label],
            "pred_label":    pred_label,
            "pred_class":    CLASS_NAMES[pred_label],
            "confidence":    confidence,
            "correct":       int(is_correct),
            "top5_classes":  "|".join(n for n, _ in top5_info),
            "top5_scores":   "|".join(f"{s:.4f}" for _, s in top5_info),
        })

        # Visualize first 24 samples (all val samples if small dataset)
        viz_path = visualize_sample(sample, scores, i, gt_label, pred_label, top5_info)
        if viz_path:
            print(f"       Saved: {viz_path}")

    # Save CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults CSV saved: {RESULTS_CSV}")

    acc = correct / len(val_data) * 100
    print("\n" + "=" * 70)
    print(f"INFERENCE COMPLETE")
    print(f"  Total samples   : {len(val_data)}")
    print(f"  Correct         : {correct}")
    print(f"  Top-1 Accuracy  : {acc:.2f}%")
    print(f"  Visualizations  : {VIZ_DIR}")
    print(f"  CSV results     : {RESULTS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=None)
    args = parser.parse_args()
    main(args.checkpoint)
