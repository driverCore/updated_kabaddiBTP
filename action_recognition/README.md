# Skeleton-Based Action Recognition (ST-GCN)

12-class kabaddi action recognition using ST-GCN on COCO 17-keypoint skeletons.

## Pipeline

```
YOLOv8 (best.pt) → Player Detection
        ↓
YOLOv8-Pose → 17 COCO Keypoints per Player
        ↓
Skeleton Sequence (M=2, T=dynamic, V=17, C=2)
        ↓
ST-GCN (MMAction2 RecognizerGCN + STGCN backbone)
        ↓
12-Class Action Prediction
```

## Action Classes

| ID | Action | ID | Action |
|----|--------|----|--------|
| 0 | ankle_hold | 6 | hand_touch |
| 1 | back_kick | 7 | lion_jump |
| 2 | block | 8 | side_kick |
| 3 | chain_tackle | 9 | thigh_hold |
| 4 | dash | 10 | toe_touch |
| 5 | dubki | 11 | waist_hold |

## Source Files

### Orchestration
- **`run_experiment.py`** — End-to-end experiment script: discovers data, creates 70/30 stratified split, generates PKL datasets, writes MMAction2 config, launches training via `tools/train.py`, runs evaluation via `tools/test.py`, produces confusion matrix and metrics.

### MMAction2 Configs
- **`configs/_base_/default_runtime.py`** — Base runtime inherited by all training configs.
- **`configs/skeleton/stgcn/kabaddi_stgcn.py`** — Kabaddi ST-GCN config v1 (batch=4, RepeatDataset×10).
- **`configs/skeleton/stgcn/kabaddi_stgcn_v2.py`** — Kabaddi ST-GCN config v2 (same architecture, refined class mapping).

### Patched MMAction2 Tools
- **`tools/train.py`** — Standard MMAction2 training entry point, patched with `np.trapz` compatibility fix for numpy≥2.0.
- **`tools/test.py`** — Standard MMAction2 testing entry point, same `np.trapz` fix.

### Custom Scripts
- **`scripts/train_kabaddi.py`** — Alternative training launcher that patches `mmcv_full_available()` to avoid broken `mmcv._ext` DLL on Windows. ST-GCN uses only pure PyTorch ops.
- **`scripts/extract_skeleton_pkl.py`** — Full skeleton extraction pipeline: YOLOv8 detection → crop → YOLOv8-Pose → map keypoints back → save annotated JPGs + coordinate JSONs + PKL files for ST-GCN.
- **`scripts/infer_kabaddi.py`** — Run inference on validation samples with skeleton visualization and top-5 predictions.
- **`scripts/evaluate_kabaddi.py`** — Comprehensive evaluation: Top-1/Top-5 accuracy, per-class precision/recall/F1, confusion matrix, bar charts.
- **`scripts/inspect_pkl.py`** — PKL format validator: checks shape `(M,T,V,2)`, label ranges, sparse keypoints.
- **`scripts/monitor_training.py`** — Real-time training dashboard: reads mmengine log files and displays live progress, loss, accuracy, ETA.
- **`scripts/generate_annotations.py`** — Simple train/val annotation list generator from video folders.

## Environment (Verified)

These exact versions were used to produce the final results:

```
Python          3.11.9
PyTorch         2.5.1+cu121
CUDA Runtime    12.1
MMAction2       1.2.0
MMCV            2.1.0
MMEngine        0.10.7
NumPy           2.4.6
SciPy           1.17.1
OpenCV          4.13.0
Ultralytics     8.4.86
TorchVision     0.20.1+cu121
GPU             NVIDIA GeForce RTX 2050
```

## Setup

```bash
# 1. Create virtual environment
python -m venv mmaction_env
mmaction_env\Scripts\activate        # Windows
# source mmaction_env/bin/activate   # Linux

# 2. Install PyTorch with CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install OpenMMLab stack (order matters)
pip install mmengine==0.10.7
pip install mmcv==2.1.0
pip install mmaction2==1.2.0

# 4. Install remaining dependencies
pip install -r requirements.txt
```

## Dataset Format

The pipeline expects skeleton data as PKL files in MMAction2's `PoseDataset` format:

```python
# Each sample in the PKL list:
{
    "frame_dir": "class_name/video_name",
    "label": int,                          # 0–11
    "img_shape": (H, W),
    "original_shape": (H, W),
    "total_frames": T,
    "keypoint": np.ndarray,                # shape (M, T, V, 2) float32
    "keypoint_score": np.ndarray           # shape (M, T, V) float32
}
# M=2 persons, V=17 COCO keypoints, C=2 (x,y)
```

Datasets are **not included**. Use `scripts/extract_skeleton_pkl.py` to generate PKL files from raw videos, or `run_experiment.py` which handles the entire pipeline including PKL generation from pre-extracted keypoint JSONs.

## Training

The exact command that produced the final results:

```bash
python run_experiment.py
```

This script:
1. Discovers skeleton data from `dhruba_dataset_processed/`
2. Creates 70/30 stratified train/test split (seed=42)
3. Generates `train.pkl` and `test.pkl`
4. Writes `training_config.py` (ST-GCN, batch=8, LR=0.05, SGD, 80 epochs)
5. Calls `tools/train.py` with the generated config
6. Runs `tools/test.py` for evaluation
7. Generates confusion matrix and classification report

Alternatively, for standalone training with a pre-existing config:

```bash
# Using patched tools/train.py
python tools/train.py configs/skeleton/stgcn/kabaddi_stgcn_v2.py

# Or using the Windows-compatible launcher
python scripts/train_kabaddi.py configs/skeleton/stgcn/kabaddi_stgcn_v2.py
```

## Inference

```bash
python scripts/infer_kabaddi.py <checkpoint.pth>
```

## Model Checkpoint

Checkpoints (`.pth` files) are **not included**. Place the trained checkpoint at:
```
work_dirs/stgcn_kabaddi/best_model.pth
```

## Notes

- **Windows compatibility**: The `np.trapz` patch in `tools/train.py` and `tools/test.py` is required because numpy≥2.0 removed `np.trapz`. The `train_kabaddi.py` script additionally patches `mmcv_full_available()` to skip broken `mmcv._ext` DLL loading on Windows.
- **MMAction2 as dependency**: The ST-GCN model (`RecognizerGCN`, `STGCN` backbone, `GCNHead`, `PoseDataset`, pipeline transforms) all come from the standard MMAction2 1.2.0 package. Only the configs and custom scripts are BTP-specific.
- **Graph layout**: Uses `layout='coco'` with `mode='stgcn_spatial'` (17 COCO keypoints).
- **Skeleton extraction**: Requires `best.pt` (YOLOv8 kabaddi detector) and `yolov8n-pose.pt` (pose estimator). The detector model is included in the `models/` directory at the repository root.
