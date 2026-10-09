# Diagnostic Case Evaluation Matrix

**Project**: Kabaddi Raider Tracking & Action Recognition (BTP)  
**Experiment**: Controlled Ground-Truth Heuristic Evaluation (E0, E2, E3, E5, E7, E8)  
**Input**: Verified Ground-Truth Player Detections (Detection Errors Excluded)  
**Scope**: Expanded 11-Case Evaluation Set (`C01` to `C11`)

---

## 1. Case Evaluation Matrix

| Case ID | Sequence | Action Class | Difficulty Phenomenon | E0 (Baseline) | E2 (Torso ROI) | E3 (Pose Align) | E5 (Colour) | E7 (SAM Mask) | E8 (Depth) |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C01** | `toe1` | `toe_touch` | Severe overlap (79.1% IoU) during reaching lunge | **P** | **P** | **S** | **F** | **S** | **P** |
| **C02** | `dubki36` | `dubki` | Low ground crouch ducking (height collapses to 97.6px) | **P** | **S** | **S** | **F** | **S** | **P** |
| **C03** | `toe21` | `toe_touch` | Fast reaching toe touch and 180° recovery sprint | **F** | **P** | **P** | **F** | **P** | **F** |
| **C04** | `dash45` | `dash` | Rapid diagonal crossing dash collision (IoU 0.427) | **P** | **F** | **F** | **P** | **F** | **P** |
| **C05** | `dash42` | `dash` | Rapid multi-defender dash collision and spatial boundary squeeze | **F** | **F** | **F** | **F** | **F** | **F** |
| **C06** | `dash5` | `dash` | High-speed opening dash tackle collision (peak IoU 0.425) | **S** | **S** | **S** | **F** | **S** | **S** |
| **C07** | `bk_48` | `back_kick` | Explosive 180° back kick rotation & backward thrust | **S** | **P** | **P** | **F** | **P** | **S** |
| **C08** | `dubki1` | `dubki` | Deep ground dive under chain with 2-frame occlusion gap | **S** | **P** | **P** | **F** | **P** | **S** |
| **C09** | `dubki10` | `dubki` | Fast ducking evasion underneath oncoming defender tackle | **P** | **P** | **P** | **F** | **P** | **P** |
| **C10** | `wh41` | `waist_hold` | Multi-defender waist hold pack capture with severe clutter | **S** | **P** | **P** | **P** | **P** | **S** |
| **C11** | `wh12` | `waist_hold` | Close-quarters 1-on-1 waist hold struggle (peak IoU 0.647) | **S** | **P** | **S** | **F** | **S** | **S** |

**Legend**:
- **S (Success)**: Raider identity continuously maintained across the full sequence without ID switches ($0$ switches).
- **P (Partial / Ambiguous)**: Raider identity temporarily fragmented or suffered exactly 1 transient ID change before recovery ($1$ switch).
- **F (Failure)**: Raider identity permanently switched to an opposing player or suffered $\ge 2$ identity switches.

---

## 2. Quantitative Diagnostic Summary

> [!IMPORTANT]
> **Diagnostic-Set Observation Disclaimer**:  
> The metrics below reflect performance specifically across the **11 diagnostic cases** designed to stress-test heuristic failure boundaries across high-speed dash collisions, multi-defender pack tackles, ground ducking collapses, occlusion gaps, and intense body grapples.

| Metric | E0 (Baseline ByteTrack) | E2 (Torso ROI) | E3 (Pose Alignment) | E5 (Hip Colour) | E7 (SAM Mask IoU) | E8 (MiDaS Relative Depth) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Successful Cases (S)** | **5** (45.5%) | **2** (18.2%) | **4** (36.4%) | **0** (0.0%) | **4** (36.4%) | **5** (45.5%) |
| **Partial / Ambiguous (P)** | **4** (36.4%) | **7** (63.6%) | **5** (45.5%) | **2** (18.2%) | **5** (45.5%) | **4** (36.4%) |
| **Failed Cases (F)** | **2** (18.2%) | **2** (18.2%) | **2** (18.2%) | **9** (81.8%) | **2** (18.2%) | **2** (18.2%) |
| **Total Raider ID Switches** | **8** | **13** | **10** | **34** | **10** | **8** |

---

## 3. High-Level Performance Takeaways

1. **E0 (ByteTrack Baseline) & E8 (MiDaS Depth) Dominate Overall Continuity**:
   - Both achieved the lowest total raider ID switches (**8** switches each) and highest success rate (**5/11**, $45.5\%$).
   - Kalman velocity buffering provided decisive advantages in **C06 (`dash5`)**, **C07 (`bk_48`)**, **C10 (`wh41`)**, and **C11 (`wh12`)**, as well as bridging the 2-frame complete occlusion gap in **C08 (`dubki1`)** with zero switches.

2. **E3 (Pose Alignment) & E7 (SAM Mask IoU) Excel on Extreme Overlap and Grappling**:
   - Both maintained **10** total switches and achieved 4 clean successes across the 11 cases.
   - Cleanly resolved severe limb-lunge overlap in **C01 (`toe1`)** ($79.1\%$ IoU), ground ducking in **C02 (`dubki36`)** ($0$ switches), opening dash tackle in **C06 (`dash5`)**, and prolonged waist hold grappling in **C11 (`wh12`)** ($0$ switches).

3. **C05 (`dash42`) Represents Universal Heuristic Boundary Failure**:
   - In **C05 (`dash42`)**, aggressive multi-defender convergence and spatial boundary squeeze ($0.484$ peak IoU) caused all six heuristics to register failure ($\ge 2$ switches), establishing a clear baseline threshold where multi-cue fusion is mandatory.

4. **E5 (Colour Sampling) Continues Severe Degradation in Cluttered Scenarios**:
   - Suffered **34** total switches across the 11 cases ($0$ successes, $9$ failures).
   - Hip/shorts color extraction fails severely when players press into defender clusters (**C05**, **C08**, **C10**, **C11**) or ground textures (**C02**, **C09**).
