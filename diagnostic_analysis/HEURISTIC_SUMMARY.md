# Qualitative Heuristic Failure-Mode Analysis

**Project**: Kabaddi Raider Tracking & Action Recognition (BTP)  
**Experiment**: Controlled Ground-Truth Heuristic Evaluation  
**Target**: Qualitative Behavior, Strengths, and Failure Boundaries of Association Cues  
**Scope**: Expanded 11-Case Benchmark Suite (`C01` to `C11`)

---

## 1. Overview

This analysis synthesizes the qualitative strengths, limitations, and failure boundaries of each tracking heuristic (**E0, E2, E3, E5, E7, E8**) based **strictly on empirical evidence from the 11 controlled diagnostic cases**:
- `C01`: `toe1` (`toe_touch`, raid) — Severe spatial overlap (79.1% IoU) during reaching lunge
- `C02`: `dubki36` (`dubki`, raid) — Low ground crouch ducking beneath defenders (height collapses to 97.6px)
- `C03`: `toe21` (`toe_touch`, raid) — Fast reaching toe touch and instantaneous 180° recovery sprint
- `C04`: `dash45` (`dash`, defend) — Rapid diagonal crossing dash collision (IoU 0.427)
- `C05`: `dash42` (`dash`, defend) — Multi-defender dash collision and spatial boundary squeeze (peak IoU 0.484)
- `C06`: `dash5` (`dash`, defend) — High-speed opening dash tackle collision (peak IoU 0.425)
- `C07`: `bk_48` (`back_kick`, raid) — Explosive 180° back kick rotation & backward thrust (peak IoU 0.461)
- `C08`: `dubki1` (`dubki`, raid) — Deep ground dive under chain with 2-frame complete occlusion gap
- `C09`: `dubki10` (`dubki`, raid) — Fast ducking evasion underneath oncoming defender tackle (IoU 0.375)
- `C10`: `wh41` (`waist_hold`, defend) — Multi-defender waist hold pack capture with severe occlusion clutter
- `C11`: `wh12` (`waist_hold`, defend) — Close-quarters 1-on-1 waist hold struggle (peak IoU 0.647)

All evaluations were performed using identical, verified Ground-Truth player bounding boxes, completely eliminating object detector errors.

---

## 2. Quantitative Diagnostic Performance Matrix

| Heuristic ID & Description | Successful Cases (S) | Partial / Ambiguous (P) | Failed Cases (F) | Total Raider ID Switches |
|---|:---:|:---:|:---:|:---:|
| **E0**: Baseline ByteTrack (Kalman + 2-Stage IoU) | **5** (45.5%) | **4** (36.4%) | **2** (18.2%) | **8** |
| **E2**: Geometrical Torso ROI Slicing | **2** (18.2%) | **7** (63.6%) | **2** (18.2%) | **13** |
| **E3**: Chest & Head Pose Alignment (Keypoints) | **4** (36.4%) | **5** (45.5%) | **2** (18.2%) | **10** |
| **E5**: Hip & Shorts HSV Colour Sampling | **0** (0.0%) | **2** (18.2%) | **9** (81.8%) | **34** |
| **E7**: MobileSAM Mask Silhouette IoU | **4** (36.4%) | **5** (45.5%) | **2** (18.2%) | **10** |
| **E8**: MiDaS Relative Monocular Depth | **5** (45.5%) | **4** (36.4%) | **2** (18.2%) | **8** |

---

## 3. Heuristic Failure-Mode Synthesis

### E0 — Baseline YOLOv8 + ByteTrack Association

#### Underlying Mechanism
Multi-object tracking combining a constant-velocity Kalman filter motion model with ByteTrack's two-stage bounding-box IoU association algorithm.

#### Observed Strengths
1. **Occlusion Bridging**: In Case C08 (`dubki1`), the raider disappears completely underneath the defenders for 2 frames (frames 104–105). ByteTrack's 30-frame track buffer and constant-velocity projection maintained raider track continuity without a single ID switch (**0 switches**), outperforming single-frame spatial heuristics.
2. **High-Speed Dash Tackle Tracking**: In Case C06 (`dash5`), Kalman momentum cleanly resolved the opening dash collision at frame 2 (**0 switches**).
3. **Waist Hold Struggle & Pack Capture**: In Cases C10 (`wh41`) and C11 (`wh12`), Kalman track prediction maintained raider continuity through multi-defender encirclement and prolonged 1-on-1 body contact (**0 switches each**).
4. **Rotational Kick Recovery**: In Case C07 (`bk_48`), constant velocity smoothed out rapid limb extension and recovery (**0 switches**).

#### Observed Weaknesses
1. **Kinematic Direction Reversals**: In Case C03 (`toe21`), an explosive reach followed by an instantaneous $180^\circ$ turn and retreat caused the velocity projection to overshoot, yielding 2 switches.
2. **Multi-Defender Boundary Squeeze**: In Case C05 (`dash42`), three defenders converged simultaneously at frame 89 ($0.484$ IoU), overwhelming bounding box IoU (2 switches).
3. **Severe Overlap Ambiguity**: In Case C01 (`toe1`), extreme $79.1\%$ IoU overlap caused temporary box confusion (1 switch).

---

### E2 — Geometrical Torso ROI Slicing

#### Underlying Mechanism
Extracts a stable vertical torso sub-region ($20\%-80\%$ of bounding-box height) to exclude dynamic limb flailing, performing greedy frame-to-frame torso IoU association without an explicit motion model.

#### Observed Strengths
1. **Stable Crawling and Ground Ducking**: In Case C02 (`dubki36`), successfully maintained the crawling raider through the ducking arc (**0 switches**).
2. **Clean Directional Dashes**: Maintained identity through the opening dash tackle in Case C06 (`dash5`, **0 switches**).

#### Observed Weaknesses
1. **Lack of Motion Momentum in Collisions**: In Cases C04 (`dash45`, 3 switches) and C05 (`dash42`, 3 switches), static greedy torso matching swapped tracks during high-speed crossings and multi-defender squeezes.
2. **Discontinuity Across Occlusions**: In Case C08 (`dubki1`), complete disappearance for 2 frames broke single-frame IoU continuity, causing a transient ID switch upon re-emergence (1 switch).
3. **Contact Clutter in Grapples**: In Cases C10 (`wh41`, 1 switch) and C11 (`wh12`, 1 switch), torso crops merged with defender torsos during waist hold contact.

---

### E3 — Chest & Head Pose Alignment

#### Underlying Mechanism
Leverages YOLOv8-pose anatomical keypoints to construct upper-body orientation vectors (nose, neck, left/right shoulders), measuring anatomical vector alignment and keypoint IoU.

#### Observed Strengths
1. **Resolving Extreme Overlap**: In Case C01 (`toe1`), despite an extreme $79.1\%$ bounding box overlap, E3 achieved **0 ID switches**. The raider's distinct horizontal reaching leg keypoints and upright chest landmarks cleanly differentiated them from the corner defender.
2. **Postural Shape Delineation in Crouches**: Maintained **0 switches** on Case C02 (`dubki36`).
3. **Close-Quarters Grappling Differentiation**: In Case C11 (`wh12`), maintained **0 switches** despite $0.647$ peak IoU; upper-body orientation vectors cleanly separated raider and defender facing angles.
4. **Opening Dash Tackle**: Flawlessly tracked Case C06 (`dash5`, **0 switches**).

#### Observed Weaknesses
1. **Keypoint Collision during Multi-Defender Squeeze**: In Case C05 (`dash42`, 3 switches) and Case C04 (`dash45`, 2 switches), overlapping upper-body landmarks confused anatomical vector matching.
2. **Transient Keypoint Occlusion**: Suffered 1 transient switch each in C03 (`toe21`), C07 (`bk_48`), C08 (`dubki1`), C09 (`dubki10`), and C10 (`wh41`).

---

### E5 — Hip & Shorts Colour Sampling

#### Underlying Mechanism
Extracts the lower torso / hip region ($45\%-100\%$ of bounding-box height), computes HSV color histograms to classify player team affiliation, and biases association toward team identities.

#### Observed Strengths
1. **Diagonal Crossing Disambiguation**: In Case C04 (`dash45`), jersey color differentiation assisted in maintaining track identity during diagonal crossings (only 1 switch).
2. **Clutter Resilience**: In Case C10 (`wh41`), team color discrimination helped maintain partial continuity in the pack (1 switch).

#### Observed Weaknesses
1. **Catastrophic Failure in Defender Clusters & Contact**: Registered **34 total switches** across the 11 cases ($0$ successes, $9$ failures):
   - In Case C08 (`dubki1`, 7 switches), ground dive under defender bodies severely contaminated the shorts crop with defender uniform pixels and mat texture.
   - In Case C02 (`dubki36`, 5 switches) and C09 (`dubki10`, 2 switches), pressing against the mat blinded the crop with court color.
   - In Case C06 (`dash5`, 4 switches) and C11 (`wh12`, 3 switches), direct body contact occluded the hip region.
   - In Case C07 (`bk_48`, 4 switches), rapid $180^\circ$ rotation degraded color histogram stability.
   - In Case C05 (`dash42`, 3 switches), multi-defender boundary squeeze thoroughly corrupted the ROI.

---

### E7 — SAM Mask Silhouette IoU

#### Underlying Mechanism
Computes pixel-accurate segmentation masks using MobileSAM within detected bounding boxes, associating tracks based on exact silhouette mask IoU.

#### Observed Strengths
1. **Pixel-Perfect Silhouette Separation**: In Case C01 (`toe1`), achieved **0 ID switches** despite $79.1\%$ bounding box overlap; the segmented silhouette cleanly isolated the raider's outstretched leg.
2. **Crouch Shape Fidelity**: Achieved **0 switches** on Case C02 (`dubki36`).
3. **Grappling Disambiguation**: Achieved **0 switches** on Case C11 (`wh12`) ($0.647$ peak IoU) and Case C06 (`dash5`).

#### Observed Weaknesses
1. **Mask Merging during High-Density Collisions**: In Cases C04 (`dash45`, 2 switches) and C05 (`dash42`, 3 switches), confluent player masks during intense physical impact triggered identity swaps.
2. **Transient Single-Frame Reset**: Suffered 1 transient switch across C03 (`toe21`), C07 (`bk_48`), C08 (`dubki1`), C09 (`dubki10`), and C10 (`wh41`).

---

### E8 — MiDaS Relative Monocular Depth

#### Underlying Mechanism
Extracts continuous depth maps via MiDaS DPT-Hybrid, estimating relative depth to separate foreground players from background players.

#### Observed Strengths
1. **Benchmark-Leading Continuity Across Contact Actions**: Tied E0 for the lowest total raider switches (**8 switches**) and highest success rate (**5/11**, $45.5\%$):
   - Flawlessly tracked Case C06 (`dash5`, **0 switches**), Case C07 (`bk_48`, **0 switches**), Case C08 (`dubki1`, **0 switches**), Case C10 (`wh41`, **0 switches**), and Case C11 (`wh12`, **0 switches**).
2. **Occlusion Immunity**: In Case C08 (`dubki1`), depth ordering successfully preserved raider track continuity across the 2-frame occlusion gap.

#### Observed Weaknesses
1. **Multi-Defender Boundary Squeeze**: In Case C05 (`dash42`, 2 switches), compressed depth disparity across converging players led to identity fragmentation.
2. **Kinematic Direction Reversals**: Inherits ByteTrack's momentum vulnerability on sudden $180^\circ$ direction reversals (Case C03: `toe21`, 2 switches).
3. **Court Planar Ambiguity**: In flat ground ducking (Case C02: `dubki36`, 1 switch; Case C09: `dubki10`, 1 switch) and lunge reach (Case C01: `toe1`, 1 switch), minimal depth disparity between player and mat results in transient switches.

---

## 4. Key Engineering Insights for Final Fusion Architecture

1. **Complementarity of E0/E8 and E3/E7**:
   - **E0 (ByteTrack)** and **E8 (MiDaS Depth)** dominate continuous motion, fast tackles, and occlusion bridging.
   - **E3 (Pose Alignment)** and **E7 (SAM Silhouette IoU)** dominate extreme spatial overlap, posture deformation, and 1-on-1 grappling contact.
2. **C05 (`dash42`) Identifies Critical Multi-Cue Boundary**:
   - No single heuristic survived Case C05 ($0.484$ IoU boundary squeeze with 3 converging defenders), proving that single-cue tracking inevitably collapses in dense pack collisions. Multi-cue fusion (Pose + SAM + Kalman Depth) is strictly necessary.
3. **E5 Color Cue Deprecation/Demotion**:
   - HSV color sampling of the lower torso is completely unreliable during kabaddi body contact, ground diving, and multi-player pack captures. It should only be used as a soft global team-consistency prior rather than a local association metric.
