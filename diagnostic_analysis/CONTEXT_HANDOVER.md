# Kabaddi BTP — Complete Project Context & Handover Document

**Project**: Kabaddi Raider Tracking & Action Recognition (BTP)  
**Workspace**: `c:\Users\aryan\Downloads\BTP`  
**Current Date**: 2026-10-09  
**Target Audience**: LLM Coding Agents / Pair Programmers  

---

## 1. Executive Summary & Objective

This project addresses the challenging computer vision problem of **continuous multi-object tracking (MOT) of raiders and defenders in broadcast Kabaddi footage**, followed by raider-centric action recognition (e.g. toe touches, dubkis, back kicks, dashes, waist holds).

### Core Tracking Challenge
Kabaddi presents extreme tracking failure modes rarely seen in standard MOT benchmarks:
1. **Severe Spatial Overlap ($>70\%$ IoU)** during lunges and tackles.
2. **Violent Postural Deformation**: Rapid ground crouching / ducking where player bounding box height collapses by $>60\%$.
3. **Kinematic Discontinuities**: Sudden $180^\circ$ direction reversals and explosive recovery sprints.
4. **Complete Temporary Occlusion**: Raiders diving underneath a cluster of defenders and disappearing from sight for multiple frames.
5. **Multi-Defender Boundary Squeezes**: 2 to 4 defenders collapsing simultaneously on the raider.

---

## 2. Experimental Benchmark Setup: Isolated Ground-Truth Heuristic Evaluation

To evaluate association heuristics in **pure isolation from YOLO detector errors**, we feed **100% verified Ground-Truth player bounding boxes** directly into six distinct tracking heuristics:

### Evaluated Heuristics (E0, E2, E3, E5, E7, E8)
- **E0 (Baseline ByteTrack)**: Constant-velocity Kalman filter motion prediction + ByteTrack 2-stage bounding-box IoU matching.
- **E2 (Torso ROI Slicing)**: Vertical slice ($20\%-80\%$ of box height) greedy IoU association without motion prediction (designed to remove limb flailing).
- **E3 (Pose Alignment)**: YOLOv8-Pose anatomical keypoints measuring head/chest orientation vectors and keypoint distance.
- **E5 (Hip/Shorts Colour)**: Lower-body ($45\%-100\%$) HSV color histogram sampling to classify and preserve team identity.
- **E7 (SAM Silhouette IoU)**: MobileSAM pixel-level segmentation mask IoU within bounding boxes.
- **E8 (MiDaS Relative Depth)**: MiDaS DPT-Hybrid monocular depth estimation integrated with ByteTrack depth ordering.

---

## 3. The 11 Diagnostic Benchmark Cases (`C01` to `C11`)

Located under `diagnostic_analysis/`:

| Case ID | Sequence | Action Split | Frames | Critical Frame | Key Physical Challenge | Peak Overlap |
|:---:|:---|:---|:---:|:---:|:---|:---:|
| **C01** | `toe1` | `raid` / `toe_touch` | 26 (25–50) | 38 | Severe overlap during reaching lunge | $79.1\%$ IoU |
| **C02** | `dubki36` | `raid` / `dubki` | 29 (55–83) | 73 | Ground crouch ducking (height collapses to $97.6$ px) | $17.9\%$ IoU |
| **C03** | `toe21` | `raid` / `toe_touch` | 31 (30–60) | 44 | Reaching toe touch with instant $180^\circ$ retreat sprint | $17.7\%$ IoU |
| **C04** | `dash45` | `defend` / `dash` | 23 (6–28) | 21 | High-speed diagonal crossing dash collision | $42.7\%$ IoU |
| **C05** | `dash42` | `defend` / `dash` | 35 (60–94) | 89 | Multi-defender dash collision & boundary squeeze | $48.4\%$ IoU |
| **C06** | `dash5` | `defend` / `dash` | 28 (1–28) | 2 | High-speed opening forward dash tackle collision | $42.5\%$ IoU |
| **C07** | `bk_48` | `raid` / `back_kick` | 18 (1–18) | 17 | Explosive $180^\circ$ back kick rotation & thrust | $46.1\%$ IoU |
| **C08** | `dubki1` | `raid` / `dubki` | 43 (89–131) | 99 | Ground dive under chain with 2-frame occlusion gap | $36.1\%$ IoU |
| **C09** | `dubki10` | `raid` / `dubki` | 39 (1–39) | 8 | Fast ducking evasion underneath oncoming tackle | $37.5\%$ IoU |
| **C10** | `wh41` | `defend` / `waist_hold` | 15 (76–90) | 90 | Multi-defender waist hold pack capture in clutter | $37.5\%$ IoU |
| **C11** | `wh12` | `defend` / `waist_hold` | 34 (1–34) | 24 | Close-quarters 1-on-1 waist hold grapple struggle | $64.7\%$ IoU |

*Note: C05 (`dash42`) and C06 (`dash5`) replaced legacy test sequences (`sidekick39` and `sidekick18`). C08 (`dubki1`) was re-evaluated on updated ground truth tracks (ID 4 = raider, ID 3 = defender).*

---

## 4. Empirical Benchmark Results

### Detailed Case Matrix (Status & ID Switches)

| Case ID | Sequence | Action | E0 (Baseline) | E2 (Torso) | E3 (Pose) | E5 (Colour) | E7 (SAM) | E8 (Depth) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C01** | `toe1` | `toe_touch` | **P** (1) | **P** (1) | **S** (0) | **F** (2) | **S** (0) | **P** (1) |
| **C02** | `dubki36` | `dubki` | **P** (1) | **S** (0) | **S** (0) | **F** (5) | **S** (0) | **P** (1) |
| **C03** | `toe21` | `toe_touch` | **F** (2) | **P** (1) | **P** (1) | **F** (2) | **P** (1) | **F** (2) |
| **C04** | `dash45` | `dash` | **P** (1) | **F** (3) | **F** (2) | **P** (1) | **F** (2) | **P** (1) |
| **C05** | `dash42` | `dash` | **F** (2) | **F** (3) | **F** (3) | **F** (3) | **F** (3) | **F** (2) |
| **C06** | `dash5` | `dash` | **S** (0) | **S** (0) | **S** (0) | **F** (4) | **S** (0) | **S** (0) |
| **C07** | `bk_48` | `back_kick` | **S** (0) | **P** (1) | **P** (1) | **F** (4) | **P** (1) | **S** (0) |
| **C08** | `dubki1` | `dubki` | **S** (0) | **P** (1) | **P** (1) | **F** (7) | **P** (1) | **S** (0) |
| **C09** | `dubki10` | `dubki` | **P** (1) | **P** (1) | **P** (1) | **F** (2) | **P** (1) | **P** (1) |
| **C10** | `wh41` | `waist_hold` | **S** (0) | **P** (1) | **P** (1) | **P** (1) | **P** (1) | **S** (0) |
| **C11** | `wh12` | `waist_hold` | **S** (0) | **P** (1) | **S** (0) | **F** (3) | **S** (0) | **S** (0) |

*Legend: **S** = Success (0 switches), **P** = Partial (1 transient switch), **F** = Failure ($\ge 2$ switches).*

### Quantitative Summary Totals

| Metric | E0 (ByteTrack) | E2 (Torso ROI) | E3 (Pose Keypoints) | E5 (Hip Colour) | E7 (SAM Mask) | E8 (MiDaS Depth) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Successes (S)** | **5** (45.5%) | **2** (18.2%) | **4** (36.4%) | **0** (0.0%) | **4** (36.4%) | **5** (45.5%) |
| **Partial (P)** | **4** (36.4%) | **7** (63.6%) | **5** (45.5%) | **2** (18.2%) | **5** (45.5%) | **4** (36.4%) |
| **Failures (F)** | **2** (18.2%) | **2** (18.2%) | **2** (18.2%) | **9** (81.8%) | **2** (18.2%) | **2** (18.2%) |
| **Total Raider ID Switches** | **8** | **13** | **10** | **34** | **10** | **8** |

---

## 5. Key Empirical Takeaways for Tracking Architecture

1. **Kinematic & Depth Dominance (E0 & E8)**:
   - Lowest total switches (**8 switches each**).
   - Kalman filter velocity projection bridges occlusions without single-frame spatial confusion (e.g., C08 `dubki1` 2-frame occlusion gap = 0 switches).
   - Relative depth prevents background defender takeover during forward tackles (C06, C10, C11).
2. **Pose & Mask Dominance on Overlap & Grappling (E3 & E7)**:
   - Upper-body anatomical vectors (E3) and MobileSAM silhouettes (E7) solve severe spatial overlap where bounding boxes fail:
     - C01 (`toe1`, $79.1\%$ IoU) $\rightarrow$ 0 switches.
     - C02 (`dubki36`, ducking crawl) $\rightarrow$ 0 switches.
     - C11 (`wh12`, $0.647$ IoU waist hold struggle) $\rightarrow$ 0 switches.
3. **Universal Failure Boundary (C05 `dash42`)**:
   - In C05, three defenders converged simultaneously at frame 89 ($0.484$ IoU). All 6 heuristics failed individually ($\ge 2$ switches). This proves that **single-cue association cannot survive multi-defender boundary squeezes**, and multi-cue fusion is mandatory.
4. **Color Cue Collapse (E5)**:
   - HSV color sampling of the lower torso failed across 9 of 11 cases (**34 total switches**). Mat texture contact, body rotations, and defender contact corrupt the color ROI. E5 should only serve as a weak global team prior, never a primary local association metric.

---

## 6. Workspace File Organization

```
c:\Users\aryan\Downloads\BTP\
├── diagnostic_analysis\
│   ├── C01\ ... C11\                  # 11 case directories
│   │   ├── tracks.json                # Currently active track file
│   │   ├── tracks_E0.json ... _E8.json# Per-heuristic track evaluations
│   │   ├── tracks_GT.json             # Ground-truth reference track
│   │   └── frames\ & frames_EX\       # Rendered annotated frames (Green=Raider, Red=Defender)
│   ├── frames\                        # Global frame store mapped per case
│   ├── generate_gt_tool_cases.py      # Automated benchmark runner and frame visualizer
│   ├── validate_original_paths.py     # Automated integrity test for all 88 track JSONs & 2,568 images
│   ├── select_heuristic.py            # CLI switcher for active heuristic across all 11 cases
│   ├── experiment_results_summary.json# Machine-readable evaluation metrics and switches
│   ├── CASE_SELECTION.md              # Detailed physical profiles of all 11 sequences
│   ├── CASE_MATRIX.md                 # Complete quantitative case evaluation matrix
│   ├── OBSERVATIONS.md                # Cue-by-cue behavioral diagnostic observations
│   ├── HEURISTIC_SUMMARY.md           # Failure-mode synthesis across heuristics
│   ├── PLAN.md                        # Master evaluation roadmap and design specs
│   └── README.md                      # Comprehensive user guide and CLI commands
├── ground_truth\
│   ├── editor_app\                    # Canonical Ground-Truth Track Editor (PySide6)
│   │   ├── main.py                    # GT editor entry point
│   │   ├── controllers\edit_ctrl.py   # Edit controller with multi-frame class propagation
│   │   ├── models\history.py          # Command stack for undo/redo
│   │   └── test_propagate_class.py    # Test suite for forward class propagation
│   └── tools\ground_truth_editor\     # Synchronized legacy wrapper (forwards to editor_app)
├── dhruba_dataset_processed\          # Original source frame sequences & GT tracks
└── test_gt_tool_cases.py              # In-application GT tool test suite across cases
```

---

## 7. Key CLI Commands for Future Agents

```powershell
# 1. Regenerate all 11 diagnostic cases, run heuristics, and produce annotated frames:
python diagnostic_analysis/generate_gt_tool_cases.py

# 2. Validate all 88 track JSON files and 2,568 source frame paths:
python diagnostic_analysis/validate_original_paths.py

# 3. Test GT Editor in-application headless loading:
python test_gt_tool_cases.py

# 4. Switch the globally active heuristic across all 11 cases:
python diagnostic_analysis/select_heuristic.py E0    # Baseline ByteTrack
python diagnostic_analysis/select_heuristic.py E3    # Pose Alignment
python diagnostic_analysis/select_heuristic.py E8    # Depth ByteTrack
python diagnostic_analysis/select_heuristic.py GT    # Ground Truth

# 5. Launch the GT Editor on any case:
python ground_truth/editor_app/main.py diagnostic_analysis/C05/tracks.json
python ground_truth/editor_app/main.py diagnostic_analysis/C08/tracks.json
```

---

## 8. Immediate Next Engineering Steps

1. **Multi-Cue Fusion Tracker Implementation**:
   - Construct a unified association pipeline combining:
     - **Kalman Motion Prediction & Buffer** (E0) for temporal continuity across occlusions.
     - **MiDaS Relative Depth Ordering** (E8) for foreground/background separation during tackles.
     - **YOLOv8-Pose Anatomical Alignment** (E3) for resolving high-overlap bounding box ambiguities ($>0.5$ IoU).
     - **MobileSAM Silhouette IoU** (E7) for precise contact delineation during grapples.
   - Target benchmark: Overcoming the universal failure boundary on **Case C05 (`dash42`)** with $<1$ ID switch.
2. **Raider-Centric Action Recognition Integration**:
   - Feed the stabilized raider trajectories and pose keypoint sequences into the downstream spatial-temporal action recognition model (e.g. ST-GCN / VideoMAE).
