"""
evaluate_kabaddi.py
===================
Comprehensive evaluation of the trained ST-GCN Kabaddi model.

Generates:
  - Top-1 / Top-5 Accuracy
  - Per-class Precision, Recall, F1-score
  - Confusion matrix (saved as PNG)
  - Full classification report
  - CSV of per-sample predictions

Usage:
  cd C:/Users/aryan/Downloads/BTP/mmaction2
  python evaluate_kabaddi.py work_dirs/stgcn_kabaddi/best_top1_acc_epoch_*.pth
  python evaluate_kabaddi.py work_dirs/stgcn_kabaddi/epoch_80.pth    (latest)
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch

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

WORK_DIR    = "work_dirs/stgcn_kabaddi"
CONFIG      = "configs/skeleton/stgcn/kabaddi_stgcn_v2.py"
VAL_PKL     = "data/kabaddi/kabaddi_val.pkl"
OUTPUT_DIR  = WORK_DIR


def find_best_checkpoint():
    """Find best or latest checkpoint automatically."""
    best = sorted([f for f in os.listdir(WORK_DIR) if "best_" in f and f.endswith(".pth")])
    if best:
        return os.path.join(WORK_DIR, best[-1])
    epochs = sorted([f for f in os.listdir(WORK_DIR) if f.startswith("epoch_") and f.endswith(".pth")])
    if epochs:
        return os.path.join(WORK_DIR, epochs[-1])
    return None


def run_evaluation(checkpoint_path):
    """Run mmaction2 test and generate full evaluation report."""
    import subprocess
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run mmaction2 test with dump-results
    results_pkl = os.path.join(OUTPUT_DIR, "test_results.pkl")
    cmd = [
        sys.executable, "tools/test.py",
        CONFIG, checkpoint_path,
        "--dump", results_pkl,
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    # Load results
    with open(results_pkl, "rb") as f:
        results = pickle.load(f)

    # Load ground truth
    with open(VAL_PKL, "rb") as f:
        val_data = pickle.load(f)

    gt_labels = np.array([s['label'] for s in val_data])

    # Extract predictions
    if isinstance(results, list) and isinstance(results[0], dict):
        pred_scores = np.array([r.get('pred_score', r.get('scores', [0]*12)) for r in results])
    else:
        pred_scores = np.array(results)

    pred_labels = np.argmax(pred_scores, axis=1)

    generate_report(gt_labels, pred_labels, pred_scores)


def generate_report(gt_labels, pred_labels, pred_scores=None):
    """Generate comprehensive evaluation metrics and save outputs."""
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_recall_fscore_support, top_k_accuracy_score
    )
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import csv

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_classes = len(CLASS_NAMES)
    n_samples = len(gt_labels)

    print("\n" + "=" * 70)
    print("PHASE 8/10 — EVALUATION REPORT")
    print("=" * 70)

    # ── Top-1 accuracy ──────────────────────────────────────────────────────
    top1 = (pred_labels == gt_labels).mean() * 100
    print(f"\nTop-1 Accuracy: {top1:.2f}%")

    # ── Top-5 accuracy ──────────────────────────────────────────────────────
    if pred_scores is not None and pred_scores.ndim == 2:
        top5 = top_k_accuracy_score(gt_labels, pred_scores, k=min(5, n_classes)) * 100
        print(f"Top-5 Accuracy: {top5:.2f}%")
    else:
        top5 = None
        print("Top-5 Accuracy: N/A (no score matrix)")

    # ── Per-class metrics ────────────────────────────────────────────────────
    print("\nPer-class Classification Report:")
    print(classification_report(gt_labels, pred_labels, target_names=CLASS_NAMES, zero_division=0))

    prec, rec, f1, support = precision_recall_fscore_support(
        gt_labels, pred_labels, average=None, zero_division=0
    )

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_title(f'ST-GCN Kabaddi — Confusion Matrix\nTop-1: {top1:.2f}%', fontsize=14)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved: {cm_path}")

    # ── Per-class accuracy bar chart ─────────────────────────────────────────
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-9) * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(CLASS_NAMES, per_class_acc, color='steelblue', edgecolor='black')
    ax.set_title(f'Per-Class Accuracy — ST-GCN Kabaddi', fontsize=14)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.axhline(top1, color='red', linestyle='--', label=f'Overall Top-1: {top1:.1f}%')
    ax.legend()
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class accuracy chart saved: {bar_path}")

    # ── Save metrics CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'ID', 'Precision', 'Recall', 'F1', 'Accuracy(%)', 'Support'])
        for i, cn in enumerate(CLASS_NAMES):
            writer.writerow([cn, i, f'{prec[i]:.4f}', f'{rec[i]:.4f}', f'{f1[i]:.4f}',
                             f'{per_class_acc[i]:.2f}', int(support[i])])
        writer.writerow([])
        writer.writerow(['OVERALL', '', '', '', '', f'{top1:.2f}', n_samples])
    print(f"Metrics CSV saved: {csv_path}")

    # ── Save full summary text ────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("ST-GCN Kabaddi — Evaluation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Top-1 Accuracy: {top1:.4f}%\n")
        if top5 is not None:
            f.write(f"Top-5 Accuracy: {top5:.4f}%\n")
        f.write(f"Total Val Samples: {n_samples}\n\n")
        f.write("Per-class Classification Report:\n")
        f.write(classification_report(gt_labels, pred_labels, target_names=CLASS_NAMES, zero_division=0))
        f.write("\nPer-class Accuracy:\n")
        for i, cn in enumerate(CLASS_NAMES):
            f.write(f"  {cn:<20}: {per_class_acc[i]:.2f}%\n")
    print(f"Full evaluation report saved: {report_path}")

    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Top-1 Accuracy  : {top1:.2f}%")
    if top5:
        print(f"  Top-5 Accuracy  : {top5:.2f}%")
    print(f"  Val samples     : {n_samples}")
    print(f"  Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ST-GCN Kabaddi model")
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to checkpoint .pth file (auto-detected if not given)")
    args = parser.parse_args()

    ckpt = args.checkpoint or find_best_checkpoint()
    if ckpt is None:
        print("ERROR: No checkpoint found in", WORK_DIR)
        print("Train first with: python tools/train.py configs/skeleton/stgcn/kabaddi_stgcn_v2.py")
        sys.exit(1)

    print(f"Using checkpoint: {ckpt}")
    run_evaluation(ckpt)
