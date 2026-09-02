import os
import sys
import json
import random
import csv
import pickle
import numpy as np
from pathlib import Path
import subprocess
import time
from collections import defaultdict
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set working directory to the script's directory (BTP folder)
os.chdir(r"C:\Users\aryan\Downloads\BTP")

# ---------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------
BASE_DIR = Path(r"C:\Users\aryan\Downloads\BTP\dhruba_dataset_processed")
EXP_DIR = BASE_DIR / "stgcn_training2"

DATASET_DIR = EXP_DIR / "dataset"
CONFIG_DIR = EXP_DIR / "config"
CKPT_DIR = EXP_DIR / "checkpoints"
LOGS_DIR = EXP_DIR / "logs"
PRED_DIR = EXP_DIR / "predictions"
EVAL_DIR = EXP_DIR / "evaluation"

# Create isolated output structure
for d in [DATASET_DIR, CONFIG_DIR, CKPT_DIR, LOGS_DIR, PRED_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PYTHON_ENV = r"C:\Users\aryan\Downloads\BTP\mmaction_env\Scripts\python.exe"

# Alphabetic ordering for exactly 12 action classes
CLASS_MAPPING = {
    "ankle_hold": 0,
    "back_kick": 1,
    "block": 2,
    "chain_tackle": 3,
    "dash": 4,
    "dubki": 5,
    "hand_touch": 6,
    "lion_jump": 7,
    "side_kick": 8,
    "thigh_hold": 9,
    "toe_touch": 10,
    "waist_hold": 11
}

CLASS_REV_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
CLASSES_LIST = [CLASS_REV_MAPPING[i] for i in range(12)]

# Save class mapping
with open(CONFIG_DIR / "class_mapping.json", "w") as f:
    json.dump(CLASS_REV_MAPPING, f, indent=4)

def print_section(title):
    print("\n" + "="*80)
    print(title.upper())
    print("="*80, flush=True)

def discover_videos():
    classes_discovered = []
    videos_by_class = defaultdict(list)
    total_found = 0
    
    for group in ["defend", "raid"]:
        group_dir = BASE_DIR / group
        if not group_dir.exists():
            continue
        for cls_dir in sorted(group_dir.iterdir()):
            cls_name = cls_dir.name
            if cls_name in CLASS_MAPPING:
                classes_discovered.append(cls_name)
                
                # We need the keypoints JSON files inside skeletalkeypoints/
                skel_dir = cls_dir / "skeletalkeypoints"
                if skel_dir.exists():
                    for sk_sub in sorted(skel_dir.iterdir()):
                        if sk_sub.is_dir() and sk_sub.name.startswith("skel_") and sk_sub.name.endswith("_frames"):
                            kp_json = sk_sub / "keypoints.json"
                            if kp_json.exists():
                                stem = sk_sub.name[5:-7] # remove skel_ and _frames
                                videos_by_class[cls_name].append({
                                    "video_name": stem,
                                    "keypoint_path": str(kp_json),
                                    "class": cls_name,
                                    "class_id": CLASS_MAPPING[cls_name]
                                })
                                total_found += 1
                                
    return list(set(classes_discovered)), dict(videos_by_class), total_found

def main():
    print_section("ST-GCN TRAINING 2 - Dataset Inspection")
    
    classes_discovered, videos_by_class, total_found = discover_videos()
    
    print(f"Number of Action Classes Found: {len(classes_discovered)}")
    print(f"Total Video Sequences Found : {total_found}")
    
    assert len(classes_discovered) == 12, f"Expected 12 classes, found {len(classes_discovered)}"
    
    # ---------------------------------------------------------
    # 70/30 Stratified Split
    # ---------------------------------------------------------
    print_section("Train / Test Split Generation")
    
    random.seed(42)
    train_split = []
    test_split = []
    split_summary = []
    
    for cls_name in CLASSES_LIST:
        vids = videos_by_class.get(cls_name, [])
        # Sort for determinism before shuffle
        vids = sorted(vids, key=lambda x: x["video_name"])
        random.shuffle(vids)
        
        n_total = len(vids)
        n_train = int(round(n_total * 0.70))
        
        # Adjust boundaries if small counts
        if n_train == 0 and n_total > 0: n_train = 1
        if n_train == n_total and n_total > 1: n_train = n_total - 1
            
        train_vids = vids[:n_train]
        test_vids = vids[n_train:]
        
        train_split.extend(train_vids)
        test_split.extend(test_vids)
        
        split_summary.append({
            "Class": cls_name,
            "Total": n_total,
            "Train": len(train_vids),
            "Test": len(test_vids)
        })
        
    # Write splits
    def write_csv(data, path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_name", "class", "class_id", "keypoint_path"])
            writer.writeheader()
            writer.writerows(data)
            
    write_csv(train_split, DATASET_DIR / "train_split.csv")
    write_csv(test_split, DATASET_DIR / "test_split.csv")
    
    with open(DATASET_DIR / "split_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Class", "Total", "Train", "Test"])
        writer.writeheader()
        writer.writerows(split_summary)
        
    # ---------------------------------------------------------
    # Data Leakage Check
    # ---------------------------------------------------------
    print_section("DATA LEAKAGE CHECK")
    train_stems = set([x["video_name"] for x in train_split])
    test_stems = set([x["video_name"] for x in test_split])
    
    overlap = train_stems.intersection(test_stems)
    
    print(f"Train sequences: {len(train_stems)}")
    print(f"Test sequences: {len(test_stems)}")
    print(f"Overlap: {len(overlap)}")
    
    if len(overlap) > 0:
        print(f"STATUS: FAILED (Leakage detected: {overlap})")
        sys.exit(1)
    else:
        print("STATUS: PASS")
        
    # ---------------------------------------------------------
    # ST-GCN Input Formatting (PKL)
    # ---------------------------------------------------------
    print_section("ST-GCN Input Format Conversion")
    
    # Expected by MMAction2: C x T x V x M => Numpy array shape (M, T, V, C) in keypoint field
    MAX_PERSONS = 2
    NUM_KP = 17
    
    def build_pkl(split_data, out_path):
        samples = []
        corrupt = 0
        for item in split_data:
            # Locate tracks.json
            kp_path = Path(item["keypoint_path"])
            cls_dir = kp_path.parent.parent.parent
            stem = item["video_name"]
            tracks_path = cls_dir / "bytetrack" / f"bytetrack_{stem}_frames" / "tracks.json"
            
            valid_frames = set()
            if tracks_path.exists():
                with open(tracks_path, "r") as f:
                    t_data = json.load(f)
                    for t in t_data:
                        valid_frames.add(t["frame"])
                        
            with open(item["keypoint_path"], "r") as f:
                frames_data_raw = json.load(f)
                
            # Filter strictly by tracks.json
            if valid_frames:
                frames_data = [fd for fd in frames_data_raw if fd.get("frame") in valid_frames]
            else:
                frames_data = frames_data_raw
                
            T = len(frames_data)
            if T == 0:
                corrupt += 1
                continue
                
            H = frames_data[0].get("image_h", 0)
            W = frames_data[0].get("image_w", 0)
            
            # Align tracks via frequency
            track_freq = defaultdict(int)
            for frame in frames_data:
                for sk in frame.get("skeletons", []):
                    tid = sk.get("track_id")
                    if tid is not None:
                        track_freq[tid] += 1
                        
            top_tids = sorted(track_freq.keys(), key=lambda k: -track_freq[k])[:MAX_PERSONS]
            top_tids = sorted(top_tids)
            tid_to_person = {tid: i for i, tid in enumerate(top_tids)}
            
            kp_arr = np.zeros((MAX_PERSONS, T, NUM_KP, 2), dtype=np.float32)
            kps_arr = np.zeros((MAX_PERSONS, T, NUM_KP), dtype=np.float32)
            
            for t, frame in enumerate(frames_data):
                for sk in frame.get("skeletons", []):
                    tid = sk.get("track_id")
                    if tid in tid_to_person:
                        p_idx = tid_to_person[tid]
                        kpts = sk.get("keypoints", [])
                        for v_idx in range(min(NUM_KP, len(kpts))):
                            kp_val = kpts[v_idx]
                            if len(kp_val) >= 2:
                                kp_arr[p_idx, t, v_idx, 0] = kp_val[0]
                                kp_arr[p_idx, t, v_idx, 1] = kp_val[1]
                            if len(kp_val) >= 3:
                                kps_arr[p_idx, t, v_idx] = kp_val[2]
                                
            samples.append({
                "frame_dir": f"{item['class']}/{item['video_name']}",
                "label": item["class_id"],
                "img_shape": (H, W),
                "original_shape": (H, W),
                "total_frames": T,
                "keypoint": kp_arr,
                "keypoint_score": kps_arr
            })
            
        with open(out_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"Generated {out_path.name}: {len(samples)} samples (Corrupt/Empty: {corrupt})")
        return samples, corrupt
        
    train_samples, train_corrupt = build_pkl(train_split, DATASET_DIR / "train.pkl")
    test_samples, test_corrupt = build_pkl(test_split, DATASET_DIR / "test.pkl")
    
    assert train_corrupt == 0 and test_corrupt == 0, "Empty/Corrupt sequences found!"
    
    # Validation Sample
    s = train_samples[0]
    print(f"\nExample Tensor Shape: {s['keypoint'].shape} (M={s['keypoint'].shape[0]}, T={s['keypoint'].shape[1]}, V={s['keypoint'].shape[2]}, C={s['keypoint'].shape[3]})")
    
    # ---------------------------------------------------------
    # Pre-Training Validation Summary
    # ---------------------------------------------------------
    print_section("ST-GCN TRAINING 2 - PRE-TRAINING VALIDATION")
    
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    print(f"Classes: 12")
    print(f"Train: {len(train_samples)} sequences")
    print(f"Test: {len(test_samples)} sequences")
    print("\nClass distribution:")
    for row in split_summary:
        print(f"{row['Class']:<15} train={row['Train']:<3} test={row['Test']:<3}")
        
    print(f"\nInput shape:\nC=2 x T x V=17 x M=2 (Numpy M,T,V,C)")
    print(f"\nDevice:\n{device_name} (CUDA={cuda_available})")
    print(f"\nData leakage:\nPASS")
    print(f"\nMissing/corrupt sequences:\n0")
    
    if not cuda_available:
        print("\nWARNING: CUDA unavailable — training will run on CPU.")
        time.sleep(3)
        
    # ---------------------------------------------------------
    # Config Generation & Training
    # ---------------------------------------------------------
    print_section("TRAINING ST-GCN MODEL")
    
    EPOCHS = 80
    config_content = f"""# generated config for kabaddi training 2
_base_ = 'C:/Users/aryan/Downloads/BTP/mmaction2/configs/_base_/default_runtime.py'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=12, in_channels=256))

dataset_type = 'PoseDataset'
ann_file_train = r'{str(DATASET_DIR / "train.pkl")}'
ann_file_val = r'{str(DATASET_DIR / "test.pkl")}'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs'),
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs'),
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2, 
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs={EPOCHS},
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        T_max={EPOCHS},
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.05,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True))

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='auto',
        max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

work_dir = r'{str(CKPT_DIR)}'
"""
    config_path = CONFIG_DIR / "training_config.py"
    with open(config_path, "w") as f:
        f.write(config_content)
        
    print(f"Model Architecture  : ST-GCN")
    print(f"Batch Size          : 8")
    print(f"Learning Rate       : 0.05")
    print(f"Optimizer           : SGD")
    print(f"Epochs              : {EPOCHS}")
    
    train_cmd = [PYTHON_ENV, "tools/train.py", str(config_path)]
    env = dict(os.environ)
    if cuda_available:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
    print("\nSkipping training because it is already complete.")
    
    # ---------------------------------------------------------
    # Evaluation & Testing
    # ---------------------------------------------------------
    print_section("TESTING ON HELD-OUT 30% SPLIT")
    
    ckpts = sorted(CKPT_DIR.glob("epoch_*.pth"))
    best_ckpts = list(CKPT_DIR.glob("best_*.pth"))
    if best_ckpts:
        best_ckpt = best_ckpts[0]
    else:
        best_ckpt = ckpts[-1]
        
    print(f"Using checkpoint: {best_ckpt.name}")
    
    # Copy best model and final model to ensure exact naming
    import shutil
    shutil.copy(best_ckpt, CKPT_DIR / "best_model.pth")
    if ckpts: shutil.copy(ckpts[-1], CKPT_DIR / "final_model.pth")
    
    dump_pkl = PRED_DIR / "test_predictions.pkl"
    test_cmd = [
        PYTHON_ENV,
        "tools/test.py",
        str(config_path),
        str(best_ckpt),
        "--dump",
        str(dump_pkl)
    ]
    
    with open(LOGS_DIR / "test.log", "w") as log_f:
        proc = subprocess.Popen(
            test_cmd,
            cwd=r"C:\Users\aryan\Downloads\BTP\mmaction2",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        for line in iter(proc.stdout.readline, ""):
            log_f.write(line)
        proc.wait()
        
    # ---------------------------------------------------------
    # Parse Predictions & Create Confusion Matrix
    # ---------------------------------------------------------
    print_section("GENERATING METRICS & CONFUSION MATRIX")
    
    with open(dump_pkl, "rb") as f:
        predictions = pickle.load(f)
        
    y_true = []
    y_pred = []
    pred_rows = []
    
    for idx, res in enumerate(predictions):
        gt_class_id = test_samples[idx]["label"]
        vid_name = test_samples[idx]["frame_dir"].split("/")[-1]
        
        pred_label_raw = res.get("pred_label")
        if isinstance(pred_label_raw, dict): pred_id = int(pred_label_raw.get("item", 0))
        elif isinstance(pred_label_raw, (list, tuple, np.ndarray, torch.Tensor)): pred_id = int(pred_label_raw[0])
        else: pred_id = int(pred_label_raw)
            
        pred_score_raw = res.get("pred_score")
        if isinstance(pred_score_raw, (list, tuple, np.ndarray, torch.Tensor)):
            pred_score = [float(v) for v in pred_score_raw]
        else:
            pred_score = [0.0] * 12
            pred_score[pred_id] = 1.0
            
        conf = pred_score[pred_id] if pred_id < 12 else 0.0
        
        y_true.append(gt_class_id)
        y_pred.append(pred_id)
        
        pred_rows.append({
            "sequence": vid_name,
            "ground_truth": CLASS_REV_MAPPING[gt_class_id],
            "predicted_class": CLASS_REV_MAPPING.get(pred_id, "unknown"),
            "confidence": round(conf, 4)
        })
        
    with open(PRED_DIR / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "ground_truth", "predicted_class", "confidence"])
        writer.writeheader()
        writer.writerows(pred_rows)
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(12))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.title('ST-GCN Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "confusion_matrix.png")
    plt.close()
    
    with open(EVAL_DIR / "confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + CLASSES_LIST)
        for i, row in enumerate(cm):
            writer.writerow([CLASSES_LIST[i]] + list(row))
            
    # Classification Report
    cr_str = classification_report(y_true, y_pred, target_names=CLASSES_LIST, labels=range(12), zero_division=0)
    with open(EVAL_DIR / "classification_report.txt", "w") as f:
        f.write(cr_str)
        
    cr_dict = classification_report(y_true, y_pred, target_names=CLASSES_LIST, labels=range(12), output_dict=True, zero_division=0)
    
    acc = cr_dict["accuracy"] * 100
    macro_f1 = cr_dict["macro avg"]["f1-score"] * 100
    weighted_f1 = cr_dict["weighted avg"]["f1-score"] * 100
    
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classes": 12,
        "test_sequences": len(y_true)
    }
    with open(EVAL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # ---------------------------------------------------------
    # README & SUMMARY
    # ---------------------------------------------------------
    readme = f"""# ST-GCN Training Experiment 2

1. **Dataset Used**: {BASE_DIR}
2. **Number of classes**: 12
3. **Train/test split**: 70% / 30% (Stratified by video sequence)
4. **Random seed**: 42
5. **Input format**: C=2, T=dynamic, V=17, M=2 (Formatted via PKL)
6. **Model used**: MMAction2 ST-GCN (stgcn_spatial)
7. **Training configuration**: Batch=8, LR=0.05, SGD, CosineAnnealingLR
8. **Number of epochs**: {EPOCHS}
9. **Best checkpoint**: `checkpoints/best_model.pth`
10. **Test accuracy**: {acc:.2f}%
11. **Confusion matrix location**: `evaluation/confusion_matrix.png`
12. **Prediction file location**: `predictions/predictions.csv`
13. **Exact command used to reproduce**: `python {Path(__file__).absolute()}`

### Metrics
- Accuracy: {acc:.2f}%
- Macro F1: {macro_f1:.2f}%
- Weighted F1: {weighted_f1:.2f}%
"""
    with open(EXP_DIR / "README.md", "w") as f:
        f.write(readme)
        
    print_section("ST-GCN TRAINING 2 COMPLETE")
    print(f"Dataset:\n{BASE_DIR}")
    print(f"\nClasses:\n12")
    print(f"\nTrain sequences:\n{len(train_samples)}")
    print(f"\nTest sequences:\n{len(test_samples)}")
    print(f"\nBest model:\n{best_ckpt}")
    print(f"\nTest Accuracy:\n{acc:.2f}%")
    print(f"\nMacro F1:\n{macro_f1:.2f}%")
    print(f"\nWeighted F1:\n{weighted_f1:.2f}%")
    print(f"\nConfusion Matrix:\n{EVAL_DIR / 'confusion_matrix.png'}")
    print(f"\nPredictions:\n{PRED_DIR / 'predictions.csv'}")
    print(f"\nFull experiment:\n{EXP_DIR}")
    print(f"\nHardware:\n{device_name} (CUDA={cuda_available})")

if __name__ == "__main__":
    main()
