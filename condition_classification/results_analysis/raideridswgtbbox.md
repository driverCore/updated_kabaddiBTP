# Raider-Only GT Controlled Evaluation

Generated: 2026-09-10 03:58

## Purpose
Isolates YOLO detection errors from tracker association errors.

## GT Source
- `dhruba_dataset_processed/raid/` (347 sequences)
- `dhruba_dataset_processed/defend/` (374 sequences)
- **Total GT files**: 721
- **Filter**: `class_name == 'raider'` only

## Part A — YOLO Raider Detection
- **Model**: `tracking/models/best.pt`
- **Settings**: conf=0.25, iou=0.50, imgsz=640
- **IoU threshold** (matching): 0.50

| Metric | Value |
|---|---:|
| GT Raider Instances | 26316 |
| YOLO Raider Detections | 117379 |
| TP | 18152 |
| FP | 99227 |
| FN | 8164 |
| Precision | 0.1546 |
| Recall | 0.6898 |
| F1 | 0.2526 |
| Mean IoU | 0.8165 |
| Median IoU | 0.8419 |
| Raider Miss Rate | 0.3102 |

## Part B — GT-BBox Tracking (E0/E2/E3/E5)
E7 and E8 intentionally skipped.

| Heuristic | Input | Raider IDSW | IDF1 | AssA | Match Rate |
|---|---|---:|---:|---:|---:|
| E0 | GT BBox | 179 | 0.9738 | 0.9931 | 98.06% |
| E2 | GT BBox | 745 | 0.9717 | 0.9717 | 100.0% |
| E3 | GT BBox | 718 | 0.9727 | 0.9727 | 100.0% |
| E5 | GT BBox | 658 | 0.6638 | 0.9637 | 68.88% |

## Output Files
```
results_analysis/YOLO_Raider_Detection_Metrics/
  per_video_results.csv         ← YOLO detection per-sequence
  detection_summary.csv         ← per-action aggregation
  comparison_with_tracking.csv  ← combined detection+tracking
  summary/
    yolo_detection_metrics.json ← global metrics JSON

results&analysis/raider_gt_controlled_experiment/
  yolo_detection/               ← raw detection JSON + CSVs
  tracking/E0,E2,E3,E5/         ← MOT CSV tracker outputs + metrics
  comparison.csv
  README.md
```

## Notes
- HOTA not computed (requires full MOTChallenge evaluation toolkit)
- IDF1 is an approximation: 2*IDTP / (2*IDTP + IDFP + IDFN)
- AssA = 1 - (IDSW / matched_frames); raider-focused
- E5 uses lower-body bbox crop fallback for colour sampling
  (no ByteTrack raw IDs available in GT-bbox mode)
- All original E0/E2/E3/E5/E7/E8 implementations unchanged