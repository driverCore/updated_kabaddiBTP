# Condition-Wise Diagnostic Classification & Analysis

This directory contains the condition-wise diagnostic classification framework, mathematical formulations, evaluation scripts, and benchmark results for evaluating multi-object tracking heuristics on Kabaddi video sequences.

---

## Purpose of Condition-Wise Classification

### Motivation

The overall tracking results provide a single summary of each heuristic's performance, but they do not explain *under which tracking conditions* a heuristic performs well or poorly. In Kabaddi, player tracking becomes particularly challenging when the Raider overlaps with defenders, moves rapidly, or changes relative depth and position during an encounter.

To investigate these effects, the diagnostic analysis classifies eligible Raider ground-truth observations into different conditions and evaluates each heuristic separately within those conditions.

### Classification Dimensions

The analysis considers three dimensions:

1. **Overlap:** Very high overlap, high overlap, and low overlap, to investigate whether the degree of bounding-box overlap between the Raider and opponents is associated with changes in identity-association performance.
2. **Speed:** High-speed and low-speed observations, to examine whether Raider movement affects the consistency of identity association.
3. **Relative depth:** Raider in front and Raider behind, to investigate whether camera-relative depth ordering is associated with differences in tracking performance.

An overall *All Eligible* category is also included to provide a reference against which condition-specific results can be compared.

### Research Objective

The objective is to determine whether different association heuristics exhibit different strengths and weaknesses under particular tracking conditions. For example, a heuristic might perform well under very high overlap but provide little benefit during low-overlap or high-speed movement. Such behaviour may be obscured by an overall aggregate score.

The analysis therefore supports three questions:

- **Condition sensitivity:** Does a heuristic's performance vary across overlap, speed, and relative-depth conditions?
- **Comparative strengths:** Which heuristics perform relatively better under each condition?
- **Failure analysis:** Are the observed weaknesses consistent with the information each heuristic uses, such as torso geometry, pose alignment, hip-region colour, segmentation masks, or depth cues?

### Evaluation Measures

The analysis reports two complementary measures:

- **Raider identity-association success rate:** The proportion of eligible Raider observations for which the associated predicted track corresponds to the correct physical Raider identity.
- **Identity switches (IDSW):** The number of identity-switch events observed under the documented evaluation protocol.

The success rate measures correctness across eligible observations, whereas IDSW measures identity-change events. They are not interchangeable: a heuristic may have more correctly associated observations while recording the same number of identity switches as another heuristic.

### Interpretation and Limitations

Condition-wise results are intended to explain *where* tracking performance differs, not merely to rank heuristics using a single aggregate number. Condition-specific findings must be interpreted alongside their denominators, the overall results, and the limitations of the evaluation.

Overlap, speed, and relative depth are distinct classification dimensions and may overlap in the same observation. Their counts should not be added together to calculate the overall population. Very-high-overlap observations may also form a separate band rather than a mutually exclusive category; the exact interpretation follows the implemented thresholds.

The analysis identifies associations between tracking conditions and observed performance. It does not, by itself, prove that a particular condition caused an identity-association failure. Any explanation of a heuristic's behaviour should be supported by the corresponding tracking outputs and diagnostic evidence.

---

## Mathematical Formulations & Classification Rules

Every eligible frame is classified along three orthogonal, independent dimensions derived strictly from ground-truth annotations and raw video frames (completely independent of heuristic tracker outputs).

### Dimension 1: Bounding-Box Overlap & Occlusion

To quantify player occlusion without relying on physical 3D contact models, 2D Intersection-over-Union ($\text{IoU}$) is computed between the Ground-Truth Raider bounding box $B_{\text{raider}}$ and each opponent bounding box $B_j$:

$$\text{IoU}(B_{\text{raider}}, B_j) = \frac{\text{Area}(B_{\text{raider}} \cap B_j)}{\text{Area}(B_{\text{raider}} \cup B_j)}$$

The frame's overlap intensity is defined by the maximum overlap across all defenders present in the frame:

$$\text{Overlap}_{\max} = \max_{j \in \text{Defenders}} \text{IoU}(B_{\text{raider}}, B_j)$$

**Classification Thresholds:**
- **Low Overlap:** $0.0 \le \text{Overlap}_{\max} \le 0.25$
- **High Overlap:** $0.25 < \text{Overlap}_{\max} \le 0.50$
- **Very High Overlap:** $\text{Overlap}_{\max} > 0.50$ *(reported as a dedicated severe-overlap band of 252 frames)*
- **Not Applicable ($\text{N/A}$):** No other ground-truth player is present in the frame ($2{,}177$ frames).

---

### Dimension 2: Raider Image-Plane Speed & Dynamics

Raider speed is measured as physical-time rate of image-plane centroid translation between consecutive ground-truth observations.

Let the bounding-box centroid $\mathbf{c}_t = (x_c, y_c)$ at frame $t$ be:

$$\mathbf{c}_t = \left(\frac{x_{\min, t} + x_{\max, t}}{2}, \; \frac{y_{\min, t} + y_{\max, t}}{2}\right)$$

Centroid velocity is calculated using forward finite difference (with backward difference fallback when $t+1$ is unavailable):

$$\Delta \mathbf{c}_t = \mathbf{c}_{t+1} - \mathbf{c}_t$$

$$v_t = \frac{\|\Delta \mathbf{c}_t\|_2}{\Delta t} = \|\mathbf{c}_{t+1} - \mathbf{c}_t\|_2 \times \text{FPS}_{\text{video}} \quad [\text{pixels / second}]$$

**Classification Thresholds:**
- **High Speed:** $v_t > 1050.0 \text{ px/s}$ *(equivalent to $> 35.0 \text{ px/frame}$ at nominal $30\text{ fps}$; $34.0\text{--}43.8\text{ px/frame}$ across video FPS $24.0\text{--}30.9$)*
- **Low Speed:** $v_t \le 1050.0 \text{ px/s}$
- **Unknown:** No adjacent valid raider observation, or video JPG count deviates from container frame count.
- **Deformation Flag (`SPEED_BOX_SHAPE_CHANGE`):** Triggered when raider box area expands or contracts by $> 50\%$ ($|\text{Area}_{t+1} - \text{Area}_t| / \text{Area}_t > 0.5$), indicating limb extension, dive, or crouch rather than pure translation.

---

### Dimension 3: Relative Depth & Camera Ordering

Relative camera depth is estimated using monocular inverse depth maps $D(x, y)$ generated by MiDaS (`MiDaS_small`), where larger values correspond to closer proximity to the camera (disparity convention).

1. **Target Opponent Selection:** The opponent $B_{\text{opp}}$ with the highest $\text{IoU} > 0$ with the Raider is chosen. If no box overlaps, the nearest opponent by center Euclidean distance within $1.0 \times H_{\text{raider}}$ (Raider box height) is selected; otherwise, Depth is marked $\text{N/A}$.
2. **Exclusive Pixel Masking:** To avoid foreground bleed into the background player's distribution, overlapping pixels are strictly excluded:
   $$\Omega_{\text{raider}} = B_{\text{raider}} \setminus (B_{\text{raider}} \cap B_{\text{opp}})$$
   $$\Omega_{\text{opp}} = B_{\text{opp}} \setminus (B_{\text{raider}} \cap B_{\text{opp}})$$
3. **Median Depth Calculation:**
   $$\tilde{d}_{\text{raider}} = \text{median}_{(x, y) \in \Omega_{\text{raider}}} D(x, y), \qquad \tilde{d}_{\text{opp}} = \text{median}_{(x, y) \in \Omega_{\text{opp}}} D(x, y)$$
4. **Robust Frame Normalisation:** To account for arbitrary per-frame MiDaS scale, difference is normalized by the frame's robust dynamic range ($95^{\text{th}}$ percentile minus $5^{\text{th}}$ percentile):
   $$\Delta_{\text{depth}} = \frac{\tilde{d}_{\text{raider}} - \tilde{d}_{\text{opp}}}{P_{95}(D) - P_5(D)}$$

**Classification Thresholds:**
- **Raider in Front:** $\Delta_{\text{depth}} > +0.05$
- **Raider Behind:** $\Delta_{\text{depth}} < -0.05$
- **Depth Unknown:** $|\Delta_{\text{depth}}| \le 0.05$ (ambiguous depth margin) or insufficient exclusive pixels ($< \max(100, 0.20 \times \text{Area})$).
- **Depth N/A:** No opponent within $1.0 \times H_{\text{raider}}$ proximity.

---

### Dimension 4: Identity Switch (IDSW) Attribution Protocol

An identity switch event occurs at frame $t$ when the predicted track ID assigned to the ground-truth Raider transitions between consecutive observed Raider frames:

$$\text{IDSW Event at frame } t \iff (\text{TrackID}_t \neq \text{TrackID}_{t-1}) \land (\text{TrackID}_t \neq \emptyset) \land (\text{TrackID}_{t-1} \neq \emptyset)$$

- **Transition Frame Attribution:** Each switch is attributed to the ground-truth condition observed at the **transition frame** $t$ (the exact frame where the switch takes effect).
- **IDSW Rate:**
  $$\text{IDSW Rate}_{\text{condition}} = \frac{N_{\text{switches}}(\text{condition})}{N_{\text{frames}}(\text{condition})} \times 100$$
- Track initialization (birth), track re-entry with the same ID, and missing tracker predictions do not increment IDSW.

---

## Condition-Wise Full Dataset Benchmark (625 Videos, 20,988 Frames)

### Table 1: Raider Identity-Association Success Rate (%)

$$\text{Success Rate} = \frac{\text{Correct Eligible Raider Frames}}{\text{Total Condition Frames}} \times 100$$

| Heuristic | Very High Overlap | High Overlap | Low Overlap | High Speed | Low Speed | Raider in Front | Raider Behind | All Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E0 (ByteTrack)** | 76.2% (192/252) | 93.2% (2203/2365) | 95.1% (15646/16446) | 92.9% (2916/3138) | 94.9% (16865/17776) | 95.6% (5643/5905) | 96.3% (5716/5933) | 94.6% (19849/20988) |
| **E2 (Torso ROI)** | 80.6% (203/252) | 90.1% (2132/2365) | 93.6% (15401/16446) | 94.2% (2955/3138) | 93.0% (16538/17776) | 92.9% (5483/5905) | 94.3% (5597/5933) | 93.2% (19556/20988) |
| **E3 (Pose Align\*)** | 84.9% (214/252) | 91.5% (2165/2365) | 94.1% (15483/16446) | 94.7% (2971/3138) | 93.6% (16639/17776) | 93.1% (5499/5905) | 94.9% (5633/5933) | 93.7% (19674/20988) |
| **E5 (Hip Colour)** | 70.2% (177/252) | 70.0% (1656/2365) | 72.9% (11992/16446) | 72.3% (2270/3138) | 72.9% (12960/17776) | 71.2% (4204/5905) | 73.6% (4364/5933) | 72.8% (15274/20988) |
| **E7 (SAM Mask)** | 60.3% (152/252) | 69.0% (1633/2365) | 85.6% (14070/16446) | 78.7% (2471/3138) | 84.3% (14981/17776) | 83.4% (4924/5905) | 84.1% (4987/5933) | 83.4% (17507/20988) |
| **E8 (MiDaS Depth)** | **89.3% (225/252)** | 88.4% (2091/2365) | 90.5% (14885/16446) | 90.2% (2832/3138) | 90.5% (16093/17776) | 87.4% (5162/5905) | 92.5% (5487/5933) | 90.5% (18991/20988) |

*\* Note: E3 = greedy full-box IoU (`associate_by_iou`); no pose keypoints.*

---

### Table 2: Condition-Wise Identity Switches (IDSW)

$$\text{Format: Switch Rate\% (Switches / Total Condition Frames)}$$

| Heuristic | Very High Overlap | High Overlap | Low Overlap | High Speed | Low Speed | Raider in Front | Raider Behind | All Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E0 (ByteTrack)** | 4.0% (10/252) | 1.6% (38/2365) | 0.6% (100/16446) | 1.2% (39/3138) | 0.6% (115/17776) | 0.6% (37/5905) | 0.5% (28/5933) | **0.7% (155/20988)** |
| **E2 (Torso ROI)** | 6.7% (17/252) | 2.1% (49/2365) | 0.6% (102/16446) | 1.3% (41/3138) | 0.7% (128/17776) | 0.7% (40/5905) | 0.7% (43/5933) | 0.8% (176/20988) |
| **E3 (Pose Align\*)** | 6.0% (15/252) | 1.6% (38/2365) | 0.6% (91/16446) | 1.1% (34/3138) | 0.6% (112/17776) | 0.6% (37/5905) | 0.6% (37/5933) | 0.7% (152/20988) |
| **E5 (Hip Colour)** | 13.1% (33/252) | 7.0% (166/2365) | 3.6% (599/16446) | 5.6% (175/3138) | 3.6% (643/17776) | 4.0% (236/5905) | 5.3% (314/5933) | 3.9% (825/20988) |
| **E7 (SAM Mask)** | 17.1% (43/252) | 7.6% (179/2365) | 2.0% (326/16446) | 4.0% (124/3138) | 2.6% (459/17776) | 2.7% (158/5905) | 2.9% (170/5933) | 2.8% (590/20988) |
| **E8 (MiDaS Depth)** | 4.0% (10/252) | 2.2% (53/2365) | 1.2% (197/16446) | 2.0% (63/3138) | 1.1% (204/17776) | 1.5% (91/5905) | 1.2% (73/5933) | 1.3% (273/20988) |

> **Analysis of E0 vs. E8 Equality in Very High Overlap (10 switches each):**
> Both E0 and E8 record exactly 10 identity switches across the 252 very-high-overlap frames. However, detailed event comparison reveals that **they are distinct tracking events**:
> - Only **4 switches** coincide on the exact same sequence and frame.
> - **6 switches** occur exclusively in E0 (where E8 tracks smoothly across severe overlap without switching).
> - **6 switches** occur exclusively in E8 (where E0 maintains continuity).
> - Their total identity-association success rates differ significantly in this condition: **E8 achieves 89.3% (225/252)** compared to **E0's 76.2% (192/252)**.

---

## Directory Organization

```
condition_classification/
├── README.md                           ← Core documentation, motivation, and mathematical formulas
├── classify_frame_conditions.py        ← Frame difficulty classification across overlap, speed, and depth
├── compute_condition_idsw.py           ← Condition-wise IDSW counter and partition verification
├── summarize_heuristic_conditions.py   ← Association accuracy and metric aggregation by condition
├── run_full_dataset_evaluation.py      ← Full-dataset evaluation execution pipeline
├── full_dataset_reports.py             ← Generates structured markdown and CSV summary reports
├── compare_heuristic_tracks.py         ← Frame-by-frame tracker comparison tool (e.g., E0 vs E8)
│
├── HEURISTIC_CONDITION_SUMMARY.md      ← Detailed tabular breakdown of all heuristics
├── heuristic_condition_summary.csv     ← CSV summary of heuristic accuracies across conditions
├── summary_cell_counts.csv             ← Frame population counts across each condition slice
├── summary_cell_counts_by_case.csv     ← Video-level condition distribution counts
├── threshold_configuration.json        ← Configured threshold parameters for classification
├── heuristic_summary_validation.json   ← Validation consistency audits
├── track_comparison_after_fix.md       ← Tracker comparison validation notes
├── track_comparison_after_fix.json     ← Tracker comparison JSON output
├── validation_report.json              ← Structural dataset validation report
├── all_cases_frame_conditions.csv      ← Frame-by-frame condition labels (20,988 frames)
├── frame_heuristic_association.csv     ← Frame-by-frame tracker predictions and matched IDs
│
└── results_analysis/                   ← Subfolder containing detailed evaluation reports
    ├── classification_summary.md       ← Comprehensive condition classification report & dataset coverage
    └── raideridswgtbbox.md             ← Part A (YOLO Raider) & Part B (GT-BBox Tracking) controlled experiment
```

---

## Verification & Replication

To verify and re-generate the condition-wise tables and IDSW counts:

```bash
# Run condition-wise IDSW tables
python condition_classification/compute_condition_idsw.py

# Run full heuristic condition summary
python condition_classification/summarize_heuristic_conditions.py
```
