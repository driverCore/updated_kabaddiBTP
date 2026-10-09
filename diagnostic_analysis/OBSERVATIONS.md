# Detailed Case Observations & Technical Interpretations

**Project**: Kabaddi Raider Tracking & Action Recognition (BTP)  
**Experiment**: Controlled Ground-Truth Heuristic Diagnostic Evaluation  
**Input**: Verified Ground-Truth Player Detections (Detection Errors Excluded)  
**Scope**: Expanded 11-Case Evaluation Set (`C01` to `C11`)

This document provides deep technical observations and cue-specific behavioral interpretations for each of the **11 diagnostic cases** evaluated across the six tracking heuristics (**E0, E2, E3, E5, E7, E8**).

---

### Case C01: `toe1`
- **Frame**: 38 (Critical Window: 35–41)
- **Action**: Toe Touch (Raid)
- **Difficulty**: Severe raider-defender overlap ($79.1\%$ IoU) during lunging extension

- **E0 (Baseline ByteTrack)**: PARTIAL (1 ID switch). Rectangular box overlap reaches $79.1\%$ as the raider lunges towards the corner defender, causing temporary box merging before recovering.
- **E2 (Torso ROI)**: PARTIAL (1 ID switch). The torso slice experiences $>70\%$ spatial overlap, causing greedy torso IoU to confuse the lunging torso with the defender.
- **E3 (Pose Alignment)**: SUCCESS (0 ID switches). Anatomical keypoint layout remains distinct; horizontal reaching leg and vertical chest landmark vectors clearly separate raider from defender.
- **E5 (Hip/Shorts Colour)**: FAILURE (2 ID switches). Lower-body crop distortion and shadow interference in the corner introduce 2 switches.
- **E7 (SAM Mask IoU)**: SUCCESS (0 ID switches). Pixel-level silhouette mask cleanly delineates the lunging raider's limb from the defending player.
- **E8 (MiDaS Relative Depth)**: PARTIAL (1 ID switch). Relative depth disparity is narrow in the corner; depth cannot fully resolve the $79.1\%$ rectangular overlap.

---

### Case C02: `dubki36`
- **Frame**: 73 (Critical Window: 70–76)
- **Action**: Dubki (Raid)
- **Difficulty**: Deep ground crawl / crouch evading high block chain (height collapses to $97.6$ px)

- **E0 (Baseline ByteTrack)**: PARTIAL (1 ID switch). Severe bounding box aspect ratio inversion ($w \gg h$) temporarily confuses Kalman velocity projection.
- **E2 (Torso ROI)**: SUCCESS (0 ID switches). Bounding box position tracks the crawling raider through the ducking arc without track swap.
- **E3 (Pose Alignment)**: SUCCESS (0 ID switches). Upper body keypoints remain stably associated across the crawl.
- **E5 (Hip/Shorts Colour)**: FAILURE (5 ID switches). Pressing the lower body directly against the court floor blinds the shorts crop with mat texture.
- **E7 (SAM Mask IoU)**: SUCCESS (0 ID switches). Silhouette mask tracks the distinct low-profile raider shape.
- **E8 (MiDaS Relative Depth)**: PARTIAL (1 ID switch). Ground plane depth proximity produces minor depth ambiguity.

---

### Case C03: `toe21`
- **Frame**: 44 (Critical Window: 41–47)
- **Action**: Toe Touch (Raid)
- **Difficulty**: Fast reaching toe touch followed by explosive $180^\circ$ reverse sprint

- **E0 (Baseline ByteTrack)**: FAILURE (2 ID switches). Abrupt direction reversal causes linear Kalman velocity extrapolation to overshoot the turning player.
- **E2 (Torso ROI)**: PARTIAL (1 ID switch). Static torso bounding box handles the turnaround with only a single transient adjustment.
- **E3 (Pose Alignment)**: PARTIAL (1 ID switch). Keypoint tracking undergoes momentary re-alignment during the rapid $180^\circ$ body flip.
- **E5 (Hip/Shorts Colour)**: FAILURE (2 ID switches). Sudden leg pivot alters jersey lighting and causes 2 switches.
- **E7 (SAM Mask IoU)**: PARTIAL (1 ID switch). Pixel silhouette undergoes quick re-association at peak turnaround.
- **E8 (MiDaS Relative Depth)**: FAILURE (2 ID switches). Depth disparity cannot overcome the kinematic overshoot.

---

### Case C04: `dash45`
- **Frame**: 21 (Critical Window: 18–24)
- **Action**: Dash (Defend)
- **Difficulty**: Rapid diagonal crossing dash collision (IoU 0.427)

- **E0 (Baseline ByteTrack)**: PARTIAL (1 ID switch). Kalman velocity vector preserves trajectory momentum through the diagonal rush.
- **E2 (Torso ROI)**: FAILURE (3 ID switches). Greedy spatial matching lacks motion inertia and swaps tracks during the crossing collision.
- **E3 (Pose Alignment)**: FAILURE (2 ID switches). Torso and shoulder keypoints intersect in 2D space during the head-on tackle.
- **E5 (Hip/Shorts Colour)**: PARTIAL (1 ID switch). Contrasting team jerseys help differentiate players across the diagonal impact.
- **E7 (SAM Mask IoU)**: FAILURE (2 ID switches). Overlapping pixel masks during physical collision cause identity confusion.
- **E8 (MiDaS Relative Depth)**: PARTIAL (1 ID switch). Depth separation aids trajectory separation despite collision contact.

---

### Case C05: `dash42`
- **Frame**: 89 (Critical Window: 85–92)
- **Action**: Dash (Defend)
- **Difficulty**: Rapid multi-defender dash collision and spatial boundary squeeze (peak IoU 0.484)

- **E0 (Baseline ByteTrack)**: FAILURE (2 ID switches). Multiple converging defenders create conflicting motion vectors that overwhelm linear Kalman extrapolation.
- **E2 (Torso ROI)**: FAILURE (3 ID switches). Concentrated perimeter tackle creates severe multi-torso overlap, triggering multiple track swaps.
- **E3 (Pose Alignment)**: FAILURE (3 ID switches). Anatomical keypoints of 3 colliding players interleave, disrupting keypoint bipartite matching.
- **E5 (Hip/Shorts Colour)**: FAILURE (3 ID switches). Multiple overlapping jerseys in a tight tackle cluster completely corrupt color extraction.
- **E7 (SAM Mask IoU)**: FAILURE (3 ID switches). Confluent pixel silhouettes blend into an indivisible multi-player mask mass.
- **E8 (MiDaS Relative Depth)**: FAILURE (2 ID switches). Relative depth disparity between players pinned against the boundary is insufficient to separate tracks.

---

### Case C06: `dash5`
- **Frame**: 2 (Critical Window: 1–6)
- **Action**: Dash (Defend)
- **Difficulty**: High-speed opening dash tackle collision (peak IoU 0.425 at frame 2)

- **E0 (Baseline ByteTrack)**: SUCCESS (0 ID switches). Unambiguous forward tackle vector is cleanly maintained without deviation.
- **E2 (Torso ROI)**: SUCCESS (0 ID switches). Upright posture during the initial impact preserves distinct torso bounding boxes.
- **E3 (Pose Alignment)**: SUCCESS (0 ID switches). Skeletal pose vectors remain well-separated across the direct charge ($0$ switches).
- **E5 (Hip/Shorts Colour)**: FAILURE (4 ID switches). Sequence onset camera pan and illumination shifts trigger 4 spurious color swaps.
- **E7 (SAM Mask IoU)**: SUCCESS (0 ID switches). Clean mask contours easily distinguish the approaching defender from the raider.
- **E8 (MiDaS Relative Depth)**: SUCCESS (0 ID switches). Marked depth gradient between the foreground tackle and court background guarantees identity continuity.

---

### Case C07: `bk_48` (Back Kick 48)
- **Frame**: 17 (Critical Window: 14–18)
- **Action**: Back Kick (Raid)
- **Difficulty**: Explosive $180^\circ$ back kick rotation and backward leg thrust (peak IoU $0.461$, height drops to $99.7$ px)

- **E0 (Baseline ByteTrack)**: SUCCESS (0 ID switches). High-momentum backward kick trajectory is cleanly tracked by ByteTrack motion model.
- **E2 (Torso ROI)**: PARTIAL (1 ID switch). Sudden body inversion from front-facing to back-facing causes 1 transient torso switch.
- **E3 (Pose Alignment)**: PARTIAL (1 ID switch). Shoulder and neck vector direction reverses $180^\circ$, causing a brief re-orientation switch.
- **E5 (Hip/Shorts Colour)**: FAILURE (4 ID switches). Rapid spinning movement into the defender boundary causes catastrophic color patch degradation (4 switches).
- **E7 (SAM Mask IoU)**: PARTIAL (1 ID switch). Silhouette tracks the rotating body shape with only 1 transient transition.
- **E8 (MiDaS Relative Depth)**: SUCCESS (0 ID switches). Raider remains distinctly positioned in depth relative to court defenders.

---

### Case C08: `dubki1`
- **Frame**: 99 (Critical Window: 95–106)
- **Action**: Dubki (Raid)
- **Difficulty**: Deep ground dive under defender chain with temporary 2-frame occlusion gap (frames 104–105)

- **E0 (Baseline ByteTrack)**: SUCCESS (0 ID switches). **Key highlight**: ByteTrack's 30-frame track buffer and constant-velocity Kalman prediction smoothly bridge the 2-frame complete occlusion gap (frames 104–105), preserving track continuity with **zero** switches!
- **E2 (Torso ROI)**: PARTIAL (1 ID switch). Torso matching tracks the ducking descent and recovers upon re-emergence with 1 transient adjustment.
- **E3 (Pose Alignment)**: PARTIAL (1 ID switch). Keypoint tracking re-aligns upon player emergence with 1 transition.
- **E5 (Hip/Shorts Colour)**: FAILURE (7 ID switches). Ground mat texture contamination and body pileup severely degrade color histograms across 7 switches.
- **E7 (SAM Mask IoU)**: PARTIAL (1 ID switch). Silhouette segmentation successfully recovers the raider after emergence with 1 transient transition.
- **E8 (MiDaS Relative Depth)**: SUCCESS (0 ID switches). Depth continuity and velocity buffering navigate the occlusion window with **zero** switches.

---

### Case C09: `dubki10`
- **Frame**: 8 (Critical Window: 6–12)
- **Action**: Dubki (Raid)
- **Difficulty**: Fast ducking evasion underneath oncoming defender tackle with aspect ratio compression (IoU 0.375)

- **E0 (Baseline ByteTrack)**: PARTIAL (1 ID switch). Kalman filter adapts to the vertical drop with only a single transient adjustment.
- **E2 (Torso ROI)**: PARTIAL (1 ID switch). Torso geometry adjusts to horizontal ducking posture with 1 transient switch.
- **E3 (Pose Alignment)**: PARTIAL (1 ID switch). Keypoints compress vertically, resolving with 1 momentary re-alignment.
- **E5 (Hip/Shorts Colour)**: FAILURE (2 ID switches). Lower-body mat occlusion produces 2 false track swaps.
- **E7 (SAM Mask IoU)**: PARTIAL (1 ID switch). Mask tracking follows the low-profile body contour with 1 transient transition.
- **E8 (MiDaS Relative Depth)**: PARTIAL (1 ID switch). Minor depth ambiguity occurs during peak head-to-head contact.

---

### Case C10: `wh41`
- **Frame**: 90 (Critical Window: 85–90)
- **Action**: Waist Hold (Defend)
- **Difficulty**: Multi-defender waist hold pack capture with severe clutter (peak IoU 0.375)

- **E0 (Baseline ByteTrack)**: SUCCESS (0 ID switches). Linear motion model cleanly tracks raider momentum into the pack with **zero** switches.
- **E2 (Torso ROI)**: PARTIAL (1 ID switch). Dense pack boundaries produce a single transient torso adjustment.
- **E3 (Pose Alignment)**: PARTIAL (1 ID switch). Skeletal pose resolves through the cluster with 1 brief re-alignment.
- **E5 (Hip/Shorts Colour)**: PARTIAL (1 ID switch). Contrasting jersey colors assist in limiting color swaps to 1 transient transition.
- **E7 (SAM Mask IoU)**: PARTIAL (1 ID switch). Multi-target silhouette cluster tracks with 1 transient transition.
- **E8 (MiDaS Relative Depth)**: SUCCESS (0 ID switches). Foreground raider depth preserves identity ahead of background pack defenders.

---

### Case C11: `wh12`
- **Frame**: 24 (Critical Window: 20–28)
- **Action**: Waist Hold (Defend)
- **Difficulty**: Close-quarters 1-on-1 waist hold struggle with extreme bounding box overlap (peak IoU 0.647)

- **E0 (Baseline ByteTrack)**: SUCCESS (0 ID switches). Constant velocity Kalman projection sustains track through the sustained grapple with **zero** switches.
- **E2 (Torso ROI)**: PARTIAL (1 ID switch). Intense body contact produces 1 transient torso switch at peak grapple.
- **E3 (Pose Alignment)**: SUCCESS (0 ID switches). Distinct pose skeletal vectors cleanly isolate the raider from the tackling defender throughout the grapple ($0$ switches).
- **E5 (Hip/Shorts Colour)**: FAILURE (3 ID switches). Intimate physical contact blends jersey color regions, causing 3 switches.
- **E7 (SAM Mask IoU)**: SUCCESS (0 ID switches). Precise SAM mask boundary delineation keeps player identities completely distinct with **zero** switches.
- **E8 (MiDaS Relative Depth)**: SUCCESS (0 ID switches). Stable relative depth ordering prevents identity swaps across the struggle.
