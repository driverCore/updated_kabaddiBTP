# Diagnostic Analysis Research Plan: Player-Tracking Heuristic Evaluation

**Project**: Kabaddi Raider Tracking & Action Recognition (BTP)  
**Experiment**: Controlled Ground-Truth Heuristic Evaluation (E0, E2, E3, E5, E7, E8)  
**Date**: October 2026  
**Status**: Approved & Active  

---

## 1. Experimental Design & Scope

### 1.1 Research Objective
The goal of this controlled experiment is to evaluate the **association and identity-tracking heuristics** in isolation from object detection errors. By providing identical, 100% clean Ground-Truth (GT) player detections to each heuristic, this experiment evaluates under what visual and motion conditions each tracking cue successfully maintains the Raider's identity, and under what conditions it fails.

```
       Ground-Truth Player Detections (All Players)
                           ↓
        Identical Input to Every Heuristic Pipeline
                           ↓
     ┌─────┬─────┬─────┬─────┬─────┬─────┐
     │ E0  │ E2  │ E3  │ E5  │ E7  │ E8  │
     └─────┴─────┴─────┴─────┴─────┴─────┘
                           ↓
         Raider Identity Association Outcome
                           ↓
      Visual Evidence (7-Panel Comparison Sheets)
```

### 1.2 Isolation & Control Rules
- **No Detector Inference**: YOLO detection inference is strictly bypassed.
- **No Confidence Filtering**: Detection bounding boxes are populated directly from verified Ground Truth with uniform confidence ($1.0$).
- **Identical Inputs**: Every heuristic processes the exact same video frame sequence, the exact same player bounding boxes, and the exact same initial Raider identification.
- **Unmodified Heuristics**: Algorithms are executed using existing repository implementations without ad-hoc parameter tuning.

---

## 2. Tracking Heuristics Under Evaluation

| Heuristic ID | Method Name | Implementation Source | Primary Association Cue | Key Visual Representation |
|:---:|:---|:---|:---|:---|
| **E0** | Baseline ByteTrack | `tracking/E0_baseline_bytetrack` | Kalman filter motion prediction + 2-stage bbox IoU | Bounding boxes + Track IDs |
| **E2** | Geometrical Torso ROI | `tracking/E2_geometrical_torso_roi` | Vertical torso slicing (20%–80% bbox height) greedy IoU | Yellow Torso ROI rectangle inside bbox |
| **E3** | Pose Alignment | `tracking/E3_chest_head_pose_alignment` | Upper-body keypoints (Head 0–4, Shoulders 5–6, Chest center) | Cyan head landmarks, green shoulders, chest axis |
| **E5** | Hip/Shorts Colour | `tracking/E5_hip_shorts_colour` | Lower-body HSV color sampling + TeamAwareIdentityTracker | Team color code (India Blue / Iran White) + Team ID |
| **E7** | SAM Mask IoU | `tracking/E7_sam_mask_iou` | MobileSAM pixel-level silhouette segmentation masks | Semi-transparent segmentation mask overlays |
| ## 3. Active Benchmark Diagnostic Cases (11 Focused Cases: C01–C11)

The active diagnostic benchmark evaluates **11 focused cases** (`C01` through `C11`) covering toe touches, dubkis, dashes, back kicks, and waist holds under verified Ground Truth detections:

| Case ID | Sequence | Action Class | Split | Total Frames | Critical Frame | Overlap (Max IoU) | Key Phenomenon & Difficulty Type |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **C01** | `toe1` | `toe_touch` | Raid | 26 (seq 25–50) | 38 | 0.791 | Severe overlap (79.1% IoU), explosive extension reach |
| **C02** | `dubki36` | `dubki` | Raid | 29 (seq 55–83) | 73 | 0.179 | Low ground crouch ducking beneath defenders (height 97.6px) |
| **C03** | `toe21` | `toe_touch` | Raid | 31 (seq 30–60) | 44 | 0.177 | Fast reaching toe touch and 180° recovery sprint |
| **C04** | `dash45` | `dash` | Defend | 23 (seq 6–28) | 21 | 0.427 | Rapid diagonal crossing dash collision (IoU 0.427) |
| **C05** | `dash42` | `dash` | Defend | 35 (seq 60–94) | 89 | 0.484 | Rapid multi-defender dash collision and spatial boundary squeeze |
| **C06** | `dash5` | `dash` | Defend | 28 (seq 1–28) | 2 | 0.425 | High-speed opening dash tackle collision (peak IoU 0.425) |
| **C07** | `bk_48` | `back_kick` | Raid | 18 (seq 1–18) | 17 | 0.461 | Explosive 180° back kick rotation and backward leg thrust |
| **C08** | `dubki1` | `dubki` | Raid | 43 (seq 89–131) | 99 | 0.361 | Ground dive under chain with 2-frame occlusion gap (frames 104–105) |
| **C09** | `dubki10` | `dubki` | Raid | 39 (seq 1–39) | 8 | 0.375 | Fast ducking evasion underneath oncoming defender tackle |
| **C10** | `wh41` | `waist_hold` | Defend | 15 (seq 76–90) | 90 | 0.375 | Multi-defender waist hold pack capture with severe occlusion clutter |
| **C11** | `wh12` | `waist_hold` | Defend | 34 (seq 1–34) | 24 | 0.647 | Close-quarters 1-on-1 waist hold struggle with extreme bounding box overlap |

---

## 4. Ground-Truth Data Verification

All 15 selected sequences have been audited:
- **Annotation files**: `dhruba_dataset_processed/[split]/[action]/bytetrack/bytetrack_[seq]_frames/tracks.json` (100% verified, valid JSON, raider labeled).
- **Frame images**: `dhruba_dataset_processed/[split]/[action]/frames/[seq]_frames/frame_*.jpg` (100% extracted frames present and readable).
- **MD5 Consistency**: Confirmed identical to `class_labelled_ground_truth`.

---

## 5. Execution Pipeline Architecture

```
                                  [tracks.json]
                                        ↓
                       Extract Frame GT Player Detections
                                        ↓
                     ┌─────────────────────────────────────┐
                     │ Parallel Heuristic Runners (E0-E8) │
                     └─────────────────────────────────────┘
                                        ↓
                        Per-Case Raider Track Matching
                                        ↓
                       Diagnostic Frame Capture Strategy:
                            [T_crit - Δt]  (Before)
                            [T_crit]       (Critical Interaction)
                            [T_crit + Δt]  (After / Resolution)
                                        ↓
                      GT Tool-Compatible Packaging:
                   [C01..C15/tracks.json] referencing ORIGINAL Source Frames
                                        ↓
                     Markdown Research Deliverables:
              CASE_SELECTION.md, CASE_MATRIX.md, OBSERVATIONS.md,
                           HEURISTIC_SUMMARY.md
```

---

## 6. Deliverable Artifacts
1. `diagnostic_analysis/PLAN.md` (this file)
2. `diagnostic_analysis/CASE_SELECTION.md`: Complete case profiles and interaction contexts
3. `diagnostic_analysis/CASE_MATRIX.md`: Evaluation matrix ($S / F / P$) across all 15 cases
4. `diagnostic_analysis/OBSERVATIONS.md`: Deep technical breakdown per case and per heuristic
5. `diagnostic_analysis/HEURISTIC_SUMMARY.md`: Qualitative failure-mode synthesis
6. `diagnostic_analysis/README.md`: Index and visual guide
7. `diagnostic_analysis/C01/tracks.json` ... `diagnostic_analysis/C15/tracks.json`: Interactive Ground Truth Tool packages referencing ORIGINAL high-quality source JPGs
8. `diagnostic_analysis/temporary_visualizations/`: Previous composite JPG sheets (demoted from primary deliverable)
