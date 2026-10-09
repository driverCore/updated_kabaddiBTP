# Controlled Ground-Truth Heuristic Evaluation Benchmark

**Project**: Kabaddi Raider Tracking & Action Recognition (BTP)  
**Experiment**: Diagnostic Evaluation of Player-Tracking Association Heuristics (E0, E2, E3, E5, E7, E8)  
**Input Data**: 100% Verified Ground-Truth Player Detections (YOLO Detector Errors Strictly Excluded)  
**Evaluation Scope**: Heuristic & Association Performance (NOT Detector Recall)  
**Primary Inspection Platform**: Existing Kabaddi Ground Truth Track Editor (`ground_truth/editor_app/`)  
**Visual Substrate**: Annotated high-quality source frame JPGs with uniform green raider / red defenders and ID switch title banners  
**Scope**: Expanded 9-Case Evaluation Set (`C01` to `C09`)

---

## 1. Overview & Experimental Motivation

In multi-object tracking for fast-paced team sports like Kabaddi, tracking failure is often conflated with object detector failure (missed detections, false positives). 

This diagnostic research benchmark completely eliminates object detection errors by feeding **verified Ground-Truth (GT) player detections** into six distinct tracking heuristics:

```
                  Verified Ground-Truth Player Detections
                                    ↓
            Identical Input Fed to All 6 Heuristics Under Test
                                    ↓
        ┌────────┬────────┬────────┬────────┬────────┬────────┐
        │   E0   │   E2   │   E3   │   E5   │   E7   │   E8   │
        └────────┴────────┴────────┴────────┴────────┴────────┘
                                    ↓
                     Raider Identity Association
                                    ↓
                   diagnostic_analysis/CXX/tracks.json
                                    ↓
                       Existing Ground Truth Tool
                                    ↓
                Annotated Source Dataset JPG Frames (CXX/frames/)
```

The experiment empirically addresses the central research question:
> *"Given correct player detections, under what visual and motion conditions does each tracking heuristic successfully maintain the Raider identity, and under what conditions does it fail?"*

---

## 2. Directory & Case Structure

Every diagnostic case (`C01` through `C09`) is packaged in its own dedicated directory containing `tracks.json` formatted directly for the existing Ground Truth Tool:

```
diagnostic_analysis/
│
├── C01/                       # Case 1: toe1 (26 frames, crit 38)
├── C02/                       # Case 2: dubki36 (29 frames, crit 73)
├── C03/                       # Case 3: toe21 (31 frames, crit 44)
├── C04/                       # Case 4: dash45 (23 frames, crit 21)
├── C05/                       # Case 5: dash42 (35 frames, crit 89)
├── C06/                       # Case 6: dash5 (28 frames, crit 2)
├── C07/                       # Case 7: bk_48 (18 frames, crit 17)
├── C08/                       # Case 8: dubki1 (43 frames, crit 99)
├── C09/                       # Case 9: dubki10 (39 frames, crit 8)
├── C10/                       # Case 10: wh41 (15 frames, crit 90)
├── C11/                       # Case 11: wh12 (34 frames, crit 24)
│   ├── tracks.json            # Primary case tracks (ByteTrack E0 baseline)
│   ├── tracks_GT.json         # Ground Truth reference tracks
│   ├── tracks_E0.json         # Heuristic E0: ByteTrack baseline
│   ├── tracks_E2.json         # Heuristic E2: Geometrical Torso ROI
│   ├── tracks_E3.json         # Heuristic E3: Chest/Head Pose Alignment
│   ├── tracks_E5.json         # Heuristic E5: Hip/Shorts Colour
│   ├── tracks_E7.json         # Heuristic E7: SAM Mask Silhouette IoU
│   ├── tracks_E8.json         # Heuristic E8: MiDaS Relative Depth
│   ├── frames/                # Active annotated frame JPGs (Raider: Green, Defenders: Red)
│   └── frames_<EX>/           # Pre-rendered annotated frame JPGs for heuristic EX
│
├── frames/                    # Master global frames directory
│   ├── C01/ .. C11/           # 321 total annotated JPG frames
│
├── PLAN.md                    # Experimental protocol & design rules
├── CASE_SELECTION.md          # Detailed profiles of all 11 selected cases
├── CASE_MATRIX.md             # Performance matrix (S / F / P) & quantitative totals
├── OBSERVATIONS.md            # Detailed cue-level observations per event
├── HEURISTIC_SUMMARY.md       # Failure-mode synthesis across all 11 cases
├── experiment_results_summary.json # Machine-readable summary metrics
├── generate_gt_tool_cases.py  # Master case generator for C01–C11
└── select_heuristic.py        # Utility to switch active tracks & frames across cases
```

---

## 3. Opening & Inspecting Cases in the Ground Truth Tool

All diagnostic cases are designed to be inspected directly through the existing GT Track Editor (`ground_truth/editor_app/`):

### Option A: Open Any Case Directly from CLI
```bash
# Open Case C01 (toe1)
python ground_truth/editor_app/main.py diagnostic_analysis/C01/tracks.json

# Open Case C05 (dash42)
python ground_truth/editor_app/main.py diagnostic_analysis/C05/tracks.json

# Open Case C07 (bk_48)
python ground_truth/editor_app/main.py diagnostic_analysis/C07/tracks.json

# Open Case C08 (dubki1)
python ground_truth/editor_app/main.py diagnostic_analysis/C08/tracks.json
```

### Option B: Open and Inspect a Specific Heuristic Variant
```bash
# Inspect Pose Alignment (E3) on Case C01 (toe1)
python ground_truth/editor_app/main.py diagnostic_analysis/C01/tracks_E3.json

# Inspect Ground Truth reference on Case C08 (dubki1)
python ground_truth/editor_app/main.py diagnostic_analysis/C08/tracks_GT.json
```

### Option C: Switch the Active Heuristic Globally
You can instantly switch `tracks.json` AND corresponding annotated frame images across all 11 cases with `select_heuristic.py`:
```bash
# Switch all cases to Torso ROI (E2)
python diagnostic_analysis/select_heuristic.py E2

# Switch all cases to Pose Alignment (E3)
python diagnostic_analysis/select_heuristic.py E3

# Switch all cases to Ground Truth reference (GT)
python diagnostic_analysis/select_heuristic.py GT

# Switch all cases back to ByteTrack baseline (E0)
python diagnostic_analysis/select_heuristic.py E0
```

### GT Tool Keyboard Shortcuts & Navigation
| Key / Shortcut | Action |
|:---|:---|
| `D` or `→` | Step forward 1 frame |
| `A` or `←` | Step backward 1 frame |
| Double-click Left | Step backward 1 frame |
| Double-click Right | Step forward 1 frame |
| `Space` | Play / pause video playback |
| `Ctrl+T` | Toggle Dark / Light theme |
| `Ctrl+O` | Open any `tracks*.json` file |

---

## 4. Visual Substrate & Labeling Guarantees

1. **Uniform Bounding Box Coloring**:
   - Raider class: Uniformly **GREEN** (`(0, 255, 0)`) with label `ID {tid} | raider` at **all frames**.
   - Defender class: Uniformly **RED** (`(0, 0, 255)`) with label `ID {tid} | defender` at **all frames**.
2. **Clear Text Title on ID Switch Frames**:
   - On the exact frame where an ID switch occurs, a prominent top banner is rendered:
     `ID SWITCH: Raider ID switched from <prev_id> -> <curr_id>  [<heuristic>]`
3. **Clean Unobstructed Display on Normal Frames**:
   - On ground truth frames or frames without identity switches, no top banner is rendered.
4. **Validation**: All 11 cases (321 frames across 88 tracking files) are verified:
   - 100% of frame files exist on disk and are verified by automated tests (`test_gt_tool_cases.py` and `validate_original_paths.py`).
   - Bounding boxes and labels are rendered directly on the native source pixels.

---

## 5. Diagnostic Case Summary (11 Cases)

| Case ID | Sequence | Action Class | Critical Frame | Diagnostic Range | Difficult Physical Phenomenon |
|:---:|:---|:---|:---:|:---:|:---|
| **C01** | `toe1` | `toe_touch` | 38 | 25–50 (26 frames) | Severe 79.1% IoU lunging toe reach |
| **C02** | `dubki36` | `dubki` | 73 | 55–83 (29 frames) | Low ground crouch ducking beneath defenders |
| **C03** | `toe21` | `toe_touch` | 44 | 30–60 (31 frames) | Fast reaching toe touch and 180° recovery sprint |
| **C04** | `dash45` | `dash` | 21 | 6–28 (23 frames) | Rapid diagonal crossing dash collision (IoU 0.427) |
| **C05** | `dash42` | `dash` | 89 | 60–94 (35 frames) | Rapid multi-defender dash collision and spatial boundary squeeze |
| **C06** | `dash5` | `dash` | 2 | 1–28 (28 frames) | High-speed opening dash tackle collision (peak IoU 0.425) |
| **C07** | `bk_48` | `back_kick` | 17 | 1–18 (18 frames) | Explosive 180° back kick rotation & backward thrust |
| **C08** | `dubki1` | `dubki` | 99 | 89–131 (43 frames) | Ground dive under chain with 2-frame occlusion gap |
| **C09** | `dubki10` | `dubki` | 8 | 1–39 (39 frames) | Fast ducking evasion underneath oncoming defender tackle |
| **C10** | `wh41` | `waist_hold` | 90 | 76–90 (15 frames) | Multi-defender waist hold pack capture with severe occlusion clutter |
| **C11** | `wh12` | `waist_hold` | 24 | 1–34 (34 frames) | Close-quarters 1-on-1 waist hold struggle (peak IoU 0.647) |

---

## 6. Heuristic Comparison Summary

| Metric | E0 (ByteTrack) | E2 (Torso ROI) | E3 (Pose Align) | E5 (Hip Colour) | E7 (SAM Mask) | E8 (Depth ByteTrack) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Successful Cases (S)** | **5** (45.5%) | **2** (18.2%) | **4** (36.4%) | **0** (0.0%) | **4** (36.4%) | **5** (45.5%) |
| **Partial / Ambiguous (P)** | **4** (36.4%) | **7** (63.6%) | **5** (45.5%) | **2** (18.2%) | **5** (45.5%) | **4** (36.4%) |
| **Failed Cases (F)** | **2** (18.2%) | **2** (18.2%) | **2** (18.2%) | **9** (81.8%) | **2** (18.2%) | **2** (18.2%) |
| **Total Raider ID Switches** | **8** | **13** | **10** | **34** | **10** | **8** |

### Key Experimental Insights
1. **Kalman Velocity (E0) and Relative Depth (E8)** achieve the highest overall continuity (**8 total switches**, **5/11 successes**). Most notably, in **C08 (`dubki1`)**, Kalman trajectory projection navigated a **2-frame complete occlusion gap** with **zero switches**, whereas static association models broke down.
2. **Pose Alignment (E3) and SAM Masks (E7)** solve **severe geometric overlap** (`toe1`, 79.1% IoU; `wh12`, 64.7% IoU) and maintained the lowest switches among spatial cues (**10 switches**).
3. **Torso ROI (E2)** works well for upright lateral movements but fails when players fold horizontally or suffer temporal occlusion gaps (**13 switches**).
4. **Colour Sampling (E5)** suffers severe degradation (**34 switches**) whenever players drop flat against the court surface or pack into dense defender clusters.
