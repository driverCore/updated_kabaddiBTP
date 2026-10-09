# Diagnostic Case Selection Catalog

**Project**: Kabaddi Raider Tracking & Action Recognition (BTP)  
**Experiment**: Controlled Ground-Truth Heuristic Diagnostic Evaluation  
**Dataset**: `dhruba_dataset_processed` (Class-labeled Ground Truth)  
**Scope**: Expanded 11-Case Evaluation Set (`C01` to `C11`)

---

## Overview

This catalog documents the **11 diagnostic cases** selected to evaluate the six player-tracking heuristics (**E0, E2, E3, E5, E7, E8**) under identical Ground-Truth (GT) player detections. 

Cases cover foundational stress tests (severe overlap, ground ducking, high-speed retreat, crossing collisions), extended defensive dash impacts, acrobatic back kicks, ground dives, and multi-defender waist hold pack tackles:

1. **C01 (`toe1`)**: Extreme raider-defender overlap ($79.1\%$ IoU) during explosive lunging extension.
2. **C02 (`dubki36`)**: Deep ground crouch ducking underneath defenders (raider height collapses to $97.6$ px).
3. **C03 (`toe21`)**: Rapid reaching foot touch and high-acceleration $180^\circ$ recovery sprint.
4. **C04 (`dash45`)**: High-velocity diagonal crossing defender dash collision ($0.427$ IoU).
5. **C05 (`dash42`)**: Rapid multi-defender dash collision and spatial boundary squeeze ($0.484$ IoU at frame 89).
6. **C06 (`dash5`)**: High-speed opening dash tackle collision ($0.425$ IoU at frame 2).
7. **C07 (`bk_48`)**: Explosive $180^\circ$ back kick rotation and backward leg thrust with peak $0.461$ IoU.
8. **C08 (`dubki1`)**: Deep ground dive/escape under defender chain with heavy contact ($0.361$ IoU) and a momentary 2-frame occlusion gap (frames 104–105).
9. **C09 (`dubki10`)**: Fast ducking evasion underneath oncoming defender tackle with bounding box aspect ratio compression ($0.375$ IoU).
10. **C10 (`wh41`)**: Multi-defender waist hold pack capture with severe occlusion clutter ($0.375$ IoU at frame 90).
11. **C11 (`wh12`)**: Close-quarters 1-on-1 waist hold struggle with extreme bounding box overlap ($0.647$ IoU at frame 24).

Detection errors are 100% eliminated by passing verified ground-truth player bounding boxes into all trackers.

---

## Summary Catalog

| Case ID | Sequence | Action Class | Split | Total Frames | Critical Frame | Overlap (Max IoU) | Primary Difficulty Phenomenon |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **C01** | `toe1` | `toe_touch` | Raid | 26 (seq 25–50) | 38 | 0.791 | Severe raider-defender overlap (79.1% IoU) during reaching lunge |
| **C02** | `dubki36` | `dubki` | Raid | 29 (seq 55–83) | 73 | 0.179 | Low ground crouch ducking beneath defenders (height collapses to 97.6px) |
| **C03** | `toe21` | `toe_touch` | Raid | 31 (seq 30–60) | 44 | 0.177 | Fast reaching toe touch and high-acceleration 180° recovery sprint |
| **C04** | `dash45` | `dash` | Defend | 23 (seq 6–28) | 21 | 0.427 | Rapid diagonal crossing dash collision (IoU 0.427) |
| **C05** | `dash42` | `dash` | Defend | 35 (seq 60–94) | 89 | 0.484 | Rapid multi-defender dash collision and spatial boundary squeeze |
| **C06** | `dash5` | `dash` | Defend | 28 (seq 1–28) | 2 | 0.425 | High-speed opening dash tackle collision (peak IoU 0.425) |
| **C07** | `bk_48` | `back_kick` | Raid | 18 (seq 1–18) | 17 | 0.461 | Explosive 180° back kick rotation and backward leg thrust (peak 0.461 IoU) |
| **C08** | `dubki1` | `dubki` | Raid | 43 (seq 89–131) | 99 | 0.361 | Deep ground dive under defender chain with 2-frame occlusion gap |
| **C09** | `dubki10` | `dubki` | Raid | 39 (seq 1–39) | 8 | 0.375 | Fast ducking evasion underneath oncoming defender tackle |
| **C10** | `wh41` | `waist_hold` | Defend | 15 (seq 76–90) | 90 | 0.375 | Multi-defender waist hold pack capture with severe occlusion clutter |
| **C11** | `wh12` | `waist_hold` | Defend | 34 (seq 1–34) | 24 | 0.647 | Close-quarters 1-on-1 waist hold struggle with extreme bounding box overlap |

---

## Detailed Case Profiles

### Case C01: `toe1`
- **Video Sequence**: `dhruba_dataset_processed/raid/toe_touch/bytetrack/bytetrack_toe1_frames/tracks.json`
- **Action Class**: `toe_touch` (Raid)
- **Frame Range**: 25–50 (Total: 26 frames, seq total 52)
- **Critical Event Frame**: 38 (Window: 35–41)
- **Difficulty Phenomenon**: Severe raider-defender overlap during lunging extension.
- **Physical Dynamics**: At frame 38, the raider performs an explosive toe touch, extending the front foot towards the right corner defender. The bounding boxes overlap with an IoU of $0.791$.
- **Why Selected**: Evaluates whether pixel-level silhouette masks (E7) or pose anatomical keypoints (E3) can distinguish players when rectangular bounding box overlap is nearly $80\%$.
- **GT Information Available**: 26 frames with continuous Raider bounding boxes.

---

### Case C02: `dubki36`
- **Video Sequence**: `dhruba_dataset_processed/raid/dubki/bytetrack/bytetrack_dubki36_frames/tracks.json`
- **Action Class**: `dubki` (Raid)
- **Frame Range**: 55–83 (Total: 29 frames, seq total 83)
- **Critical Event Frame**: 73 (Window: 70–76)
- **Difficulty Phenomenon**: Deep ground crawl / crouch evading high block chain.
- **Physical Dynamics**: Raider drops flat to the mat to slip under defenders, compressing bounding box height to just $97.6$ px at frame 73.
- **Why Selected**: Stresses color sampling (E5) and geometrical torso ROI (E2) where the upright human assumption breaks down completely.
- **GT Information Available**: 29 frames with verified annotations.

---

### Case C03: `toe21`
- **Video Sequence**: `dhruba_dataset_processed/raid/toe_touch/bytetrack/bytetrack_toe21_frames/tracks.json`
- **Action Class**: `toe_touch` (Raid)
- **Frame Range**: 30–60 (Total: 31 frames, seq total 60)
- **Critical Event Frame**: 44 (Window: 41–47)
- **Difficulty Phenomenon**: Fast reaching toe touch followed by explosive reverse sprint.
- **Physical Dynamics**: High spatial acceleration and direction reversal right at the boundary line.
- **Why Selected**: Tests Kalman velocity smoothing response when motion direction abruptly reverses $180^\circ$.
- **GT Information Available**: 31 frames with verified raider track.

---

### Case C04: `dash45`
- **Video Sequence**: `dhruba_dataset_processed/defend/dash/bytetrack/bytetrack_dash45_frames/tracks.json`
- **Action Class**: `dash` (Defend)
- **Frame Range**: 6–28 (Total: 23 frames, seq total 31)
- **Critical Event Frame**: 21 (Window: 18–24)
- **Difficulty Phenomenon**: Rapid diagonal crossing dash collision (IoU 0.427).
- **Physical Dynamics**: Defender charges diagonally across the court at high speed, colliding with raider and driving towards lobby boundary.
- **Why Selected**: Tests trajectory intersection tracking and team jersey colour consistency (E5) during high-velocity crossing trajectories.
- **GT Information Available**: 23 frames with verified labels.

---

### Case C05: `dash42`
- **Video Sequence**: `dhruba_dataset_processed/defend/dash/bytetrack/bytetrack_dash42_frames/tracks.json`
- **Action Class**: `dash` (Defend)
- **Frame Range**: 60–94 (Total: 35 frames, seq total 35)
- **Critical Event Frame**: 89 (Window: 85–92)
- **Difficulty Phenomenon**: Rapid multi-defender dash collision and spatial boundary squeeze.
- **Physical Dynamics**: Multiple defenders converge on the raider at high acceleration, pinning the raider against the boundary line. At frame 89, raider-defender IoU reaches $0.484$ in an extremely tight spatial collision.
- **Why Selected**: Tests tracking behavior under aggressive multi-defender convergence and severe spatial compression where all baseline heuristics face catastrophic bounding-box overlap.
- **GT Information Available**: 35 frames with verified labels.

---

### Case C06: `dash5`
- **Video Sequence**: `dhruba_dataset_processed/defend/dash/bytetrack/bytetrack_dash5_frames/tracks.json`
- **Action Class**: `dash` (Defend)
- **Frame Range**: 1–28 (Total: 28 frames, seq total 28)
- **Critical Event Frame**: 2 (Window: 1–6)
- **Difficulty Phenomenon**: High-speed opening dash tackle collision.
- **Physical Dynamics**: Immediately from the opening frames, a defender initiates a rapid charging tackle across the court, generating an initial peak IoU of $0.425$ at frame 2 before driving the raider out.
- **Why Selected**: Tests association robustness right at sequence onset before Kalman track filters or temporal buffers have accumulated long history.
- **GT Information Available**: 28 frames with verified tracks.

---

### Case C07: `bk_48` (Back Kick 48)
- **Video Sequence**: `dhruba_dataset_processed/raid/back_kick/bytetrack/bytetrack_bk_48_frames/tracks.json`
- **Action Class**: `back_kick` (Raid)
- **Frame Range**: 1–18 (Total: 18 frames, seq total 18)
- **Critical Event Frame**: 17 (Window: 14–18)
- **Difficulty Phenomenon**: Explosive $180^\circ$ back kick rotation and backward leg thrust.
- **Physical Dynamics**: Raider abruptly turns back to defenders and thrusts kicking foot backward; IoU spikes to $0.461$ while raider height compresses to $99.7$ px.
- **Why Selected**: Tests rotation tolerance in torso and color models; backward orientation completely reverses torso anterior/posterior appearance.
- **GT Information Available**: 18 frames with verified annotations.

---

### Case C08: `dubki1`
- **Video Sequence**: `dhruba_dataset_processed/raid/dubki/bytetrack/bytetrack_dubki1_frames/tracks.json`
- **Action Class**: `dubki` (Raid)
- **Frame Range**: 89–131 (Total: 43 frames, seq total 43)
- **Critical Event Frame**: 99 (Window: 95–106)
- **Difficulty Phenomenon**: Ground dive under defender chain with temporary 2-frame occlusion gap (frames 104–105).
- **Physical Dynamics**: Raider dives beneath two converging defenders (IoU $0.361$ at frame 99), disappears completely under the body cluster at frames 104–105, and re-emerges at frame 106. Ground-truth verified with player ID 4 as raider and ID 3 as defender.
- **Why Selected**: Tests identity re-identification across full occlusion gaps; evaluates Kalman track buffering vs static single-frame association.
- **GT Information Available**: 41 frames with verified raider bounding boxes (2 occlusion frames).

---

### Case C09: `dubki10`
- **Video Sequence**: `dhruba_dataset_processed/raid/dubki/bytetrack/bytetrack_dubki10_frames/tracks.json`
- **Action Class**: `dubki` (Raid)
- **Frame Range**: 1–39 (Total: 39 frames, seq total 39)
- **Critical Event Frame**: 8 (Window: 6–12)
- **Difficulty Phenomenon**: Fast ducking evasion underneath oncoming defender tackle with aspect ratio compression.
- **Physical Dynamics**: Rapid frontal tackle encounter where defender attempts an upper-body block while raider ducks immediately down, reaching IoU $0.375$ at frame 8.
- **Why Selected**: Tests interaction during rapid vertical dropping motion while defender moves forward aggressively.
- **GT Information Available**: 39 frames with verified tracks.

---

### Case C10: `wh41`
- **Video Sequence**: `dhruba_dataset_processed/defend/waist_hold/bytetrack/bytetrack_wh41_frames/tracks.json`
- **Action Class**: `waist_hold` (Defend)
- **Frame Range**: 76–90 (Total: 15 frames, seq total 15)
- **Critical Event Frame**: 90 (Window: 85–90)
- **Difficulty Phenomenon**: Multi-defender waist hold pack capture with severe occlusion clutter.
- **Physical Dynamics**: Pack of 7 defenders wraps around the raider in a concentrated cluster, resulting in high overlap ($0.375$ IoU at frame 90) and heavy mutual visual occlusion.
- **Why Selected**: Tests tracking resilience in dense multi-target defensive clusters where spatial boundaries and appearance features interleave.
- **GT Information Available**: 15 frames with verified labels.

---

### Case C11: `wh12`
- **Video Sequence**: `dhruba_dataset_processed/defend/waist_hold/bytetrack/bytetrack_wh12_frames/tracks.json`
- **Action Class**: `waist_hold` (Defend)
- **Frame Range**: 1–34 (Total: 34 frames, seq total 34)
- **Critical Event Frame**: 24 (Window: 20–28)
- **Difficulty Phenomenon**: Close-quarters 1-on-1 waist hold struggle with extreme bounding box overlap.
- **Physical Dynamics**: Defender secures a tight waist hold grip; the two players grapple continuously across 10+ frames with peak overlap reaching $0.647$ IoU at frame 24.
- **Why Selected**: Tests fine-grained spatial and keypoint separation under intense, prolonged body contact.
- **GT Information Available**: 34 frames with verified annotations.
