# Frame-Condition Classification

Principal population: **full dataset** — every sequence in `dhruba_dataset_processed` with reviewed GT tracks, each over its full GT frame range. The earlier selected-case (C01–C19 diagnostic) results are kept at the end of this file and in `condition_classification/C*/`.

**Depth factor status:** computed with the existing MiDaS_small implementation (`tracking/E8_midas_depth/depth_features.py`).

## Dataset coverage

- Sequences with a GT `tracks.json`: 721. **Included (reviewed GT schema): 625.** Excluded: 96 sequences in the raw-tracker schema (records without `frame_idx`, boxes without `status`/`original_id`, with `bbox_yolo` and detector confidences): never opened in the GT editor, so they are not ground truth. Excluded by class: defend/ankle_hold: 68, defend/block: 2, defend/waist_hold: 1, raid/back_kick: 25.
- Frames in the included GT ranges: 22973; **eligible raider frames: 20988** (625 videos). Ineligible: NO_RAIDER_GT_BOX: 1881, AMBIGUOUS_DUPLICATE_RAIDER_GT: 12, NO_GT_RECORD: 92.

## Frames per action class

| Class | Videos | Frames | Eligible | Very high overlap | HIGH_OVERLAP | LOW_OVERLAP | HIGH_SPEED | LOW_SPEED | FRONT | BEHIND | Depth UNKNOWN | Depth N/A (reason) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| defend/ankle_hold | 53 | 2083 | 1746 | 2 | 58 | 1621 | 441 | 1305 | 558 | 531 | 375 | 282 (217 no opponent within 1 raider-height; 65 no other player) |
| defend/block | 43 | 935 | 933 | 11 | 191 | 731 | 118 | 814 | 316 | 316 | 264 | 37 (37 no opponent within 1 raider-height; 0 no other player) |
| defend/chain_tackle | 59 | 1413 | 1332 | 42 | 459 | 831 | 204 | 1097 | 399 | 462 | 453 | 18 (18 no opponent within 1 raider-height; 0 no other player) |
| defend/dash | 52 | 1923 | 1821 | 41 | 548 | 1232 | 284 | 1537 | 485 | 802 | 467 | 67 (67 no opponent within 1 raider-height; 0 no other player) |
| defend/thigh_hold | 52 | 1451 | 1446 | 10 | 222 | 1212 | 157 | 1289 | 348 | 543 | 448 | 107 (105 no opponent within 1 raider-height; 2 no other player) |
| defend/waist_hold | 44 | 1797 | 1787 | 94 | 453 | 1228 | 161 | 1626 | 608 | 501 | 563 | 115 (103 no opponent within 1 raider-height; 12 no other player) |
| raid/back_kick | 63 | 2847 | 2049 | 6 | 51 | 1269 | 174 | 1867 | 504 | 351 | 162 | 1032 (309 no opponent within 1 raider-height; 723 no other player) |
| raid/dubki | 50 | 2842 | 2449 | 23 | 199 | 1913 | 413 | 2028 | 720 | 531 | 382 | 816 (502 no opponent within 1 raider-height; 314 no other player) |
| raid/hand_touch | 46 | 1361 | 1330 | 0 | 33 | 1095 | 213 | 1093 | 351 | 412 | 101 | 466 (264 no opponent within 1 raider-height; 202 no other player) |
| raid/lion_jump | 56 | 1882 | 1817 | 2 | 47 | 1528 | 414 | 1401 | 351 | 496 | 230 | 740 (500 no opponent within 1 raider-height; 240 no other player) |
| raid/side_kick | 60 | 1810 | 1725 | 5 | 41 | 1585 | 258 | 1467 | 465 | 559 | 233 | 468 (374 no opponent within 1 raider-height; 94 no other player) |
| raid/toe_touch | 47 | 2629 | 2553 | 16 | 63 | 2201 | 301 | 2252 | 800 | 429 | 621 | 703 (430 no opponent within 1 raider-height; 273 no other player) |
| **Total** | **625** | **22973** | **20988** | **252** | **2365** | **16446** | **3138** | **17776** | **5905** | **5933** | **4299** | **4851** (2926 / 1925) |

Overlap is NOT_APPLICABLE for 2177 eligible frames with no other GT player in the frame; they are excluded from both overlap columns (HIGH + LOW + N/A = Eligible).

Depth N/A = no eligible comparison opponent: no GT box overlaps the raider and the nearest player's centre is more than 1.0 raider box-height away (or no other player). Speed UNKNOWN frames (not shown as a column): 74 — no adjacent raider observation, or the video's JPG count differs from its frame count (FPS not trusted). Per-video counts: `summary_cell_counts_by_case.csv`; per-frame: `all_cases_frame_conditions.csv`.

## Reliability flags (eligible frames)

| Flag | Frames |
|---|---:|
| `DEPTH_DIFF_WITHIN_MARGIN` | 3520 |
| `DEPTH_INSUFFICIENT_EXCLUSIVE_PIXELS` | 779 |
| `DEPTH_NO_OPPONENT_WITHIN_PROXIMITY` | 2926 |
| `DUPLICATE_RAIDER_GT_RESOLVED_BY_CONTINUITY` | 354 |
| `RAIDER_GT_STATUS_unreviewed` | 9505 |
| `RAIDER_GT_STATUS_verified` | 1 |
| `SPEED_BACKWARD_DIFFERENCE` | 733 |
| `SPEED_BOX_SHAPE_CHANGE` | 584 |
| `SPEED_FPS_UNRELIABLE` | 49 |
| `SPEED_NO_ADJACENT_RAIDER_OBSERVATION` | 25 |

## Thresholds and rationale (unchanged from the validated experiment)

- **Overlap:** LOW, HIGH, and VERY HIGH are mutually exclusive conditions.
- **Speed:** HIGH_SPEED if speed > **1050.0 px/s**. Reused from analyze_switch_reasons.py (`velocity > 35.0` px/frame, GT box-centre displacement between consecutive frames), converted to px/s at a nominal 30 fps (35 x 30 = 1050 px/s) so the threshold is a fixed physical-time rate. At the actual case FPS (24.0-30.9) this equals 34.0-43.8 px/frame. Δt = 1/FPS of the source video; velocity only between consecutive GT raider observations.
- **Depth:** FRONT/BEHIND if |normalised median-depth difference| > **0.05**. PROPOSED, not previously established in the repository. |normalised difference| <= margin -> UNKNOWN. Sensitivity at 0.02 and 0.10 is reported in classification_summary.md.
- **Very high overlap:** separate band in the overlap partition, reported in the tables only.

## Validation

- JPG ↔ video frame correspondence: 22873/22881 JPGs closest to their own decoded video frame; 8 flagged.
- Videos whose JPG count ≠ video frame count (speed set UNKNOWN): 2 (defend/chain_tackle/ct1, raid/hand_touch/ht10).
- Videos with duplicate frame numbers: 0; with frame_idx ≠ frame−1: 0.
- `only_requested_cases`: True
- `all_rows_have_existing_original_jpg`: True
- `original_jpgs_outside_diagnostic_analysis`: True
- `iou_in_0_1`: True
- `speed_finite_nonnegative`: True
- `depth_labels_follow_inverse_depth_convention`: True
- `ineligible_rows_have_no_factor_labels`: True
- `convention_rho_frames`: 13252
- `convention_rho_positive_frames`: 12235
- `convention_rho_median`: 0.8

## Limitations

- Overlap is 2D box IoU, not physical occlusion. Speed is image-plane px/s (camera motion, resolution and perspective affect it). Depth is monocular relative inverse depth; the 0.05 margin is a proposed value. Many GT boxes (mostly defenders) are `unreviewed`. Consecutive frames are correlated.

## Raider identity-association success rate by condition — full dataset

Copied from `HEURISTIC_CONDITION_SUMMARY.md` (Part 1). Matched-ID (IDF1-style) success; counts in brackets.

**Raider ID association accuracy** = correctly associated eligible raider frames / eligible raider frames in the condition × 100. For each video and heuristic, the matched ID is the ID appearing on the GT raider box most often; a frame is correct only when that ID is on the raider box and on no other box. Missing IDs, switched/re-born IDs, and IDs shared with another player are incorrect.

| Heuristic | Very high overlap | High overlap | Low overlap | High speed | Low speed | Raider in front | Raider behind | All eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 (ByteTrack) | 76.2% (192/252) | 93.2% (2203/2365) | 95.1% (15646/16446) | 92.9% (2916/3138) | 94.9% (16865/17776) | 95.6% (5643/5905) | 96.3% (5716/5933) | 94.6% (19849/20988) |
| E2 (Torso ROI) | 80.6% (203/252) | 90.1% (2132/2365) | 93.6% (15401/16446) | 94.2% (2955/3138) | 93.0% (16538/17776) | 92.9% (5483/5905) | 94.3% (5597/5933) | 93.2% (19556/20988) |
| E3 (Pose Align*) | 84.9% (214/252) | 91.5% (2165/2365) | 94.1% (15483/16446) | 94.7% (2971/3138) | 93.6% (16639/17776) | 93.1% (5499/5905) | 94.9% (5633/5933) | 93.7% (19674/20988) |
| E5 (Hip Colour) | 70.2% (177/252) | 70.0% (1656/2365) | 72.9% (11992/16446) | 72.3% (2270/3138) | 72.9% (12960/17776) | 71.2% (4204/5905) | 73.6% (4364/5933) | 72.8% (15274/20988) |
| E7 (SAM Mask) | 60.3% (152/252) | 69.0% (1633/2365) | 85.6% (14070/16446) | 78.7% (2471/3138) | 84.3% (14981/17776) | 83.4% (4924/5905) | 84.1% (4987/5933) | 83.4% (17507/20988) |
| E8 (MiDaS Depth) | 89.3% (225/252) | 88.4% (2091/2365) | 90.5% (14885/16446) | 90.2% (2832/3138) | 90.5% (16093/17776) | 87.4% (5162/5905) | 92.5% (5487/5933) | 90.5% (18991/20988) |

\* E3 = greedy full-box IoU (associate_by_iou); no pose keypoints.

Very high overlap is a separate overlap band (252 frames from 87 videos).

### Condition-Wise Identity Switches (IDSW)

**Raider ID switch rate (IDSW rate)** = switch transition frames / eligible raider frames in the condition × 100; counts in brackets `(IDSW / frames)`. For each video and heuristic, an ID switch occurs on a frame where the predicted track ID on the ground-truth raider changes from the previous valid raider observation.

| Heuristic | Very high overlap | High overlap | Low overlap | High speed | Low speed | Raider in front | Raider behind | All eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 (ByteTrack) | 4.0% (10/252) | 1.6% (38/2365) | 0.6% (100/16446) | 1.2% (39/3138) | 0.6% (115/17776) | 0.6% (37/5905) | 0.5% (28/5933) | 0.7% (155/20988) |
| E2 (Torso ROI) | 6.7% (17/252) | 2.1% (49/2365) | 0.6% (102/16446) | 1.3% (41/3138) | 0.7% (128/17776) | 0.7% (40/5905) | 0.7% (43/5933) | 0.8% (176/20988) |
| E3 (Pose Align*) | 6.0% (15/252) | 1.6% (38/2365) | 0.6% (91/16446) | 1.1% (34/3138) | 0.6% (112/17776) | 0.6% (37/5905) | 0.6% (37/5933) | 0.7% (152/20988) |
| E5 (Hip Colour) | 13.1% (33/252) | 7.0% (166/2365) | 3.6% (599/16446) | 5.6% (175/3138) | 3.6% (643/17776) | 4.0% (236/5905) | 5.3% (314/5933) | 3.9% (825/20988) |
| E7 (SAM Mask) | 17.1% (43/252) | 7.6% (179/2365) | 2.0% (326/16446) | 4.0% (124/3138) | 2.6% (459/17776) | 2.7% (158/5905) | 2.9% (170/5933) | 2.8% (590/20988) |
| E8 (MiDaS Depth) | 4.0% (10/252) | 2.2% (53/2365) | 1.2% (197/16446) | 2.0% (63/3138) | 1.1% (204/17776) | 1.5% (91/5905) | 1.2% (73/5933) | 1.3% (273/20988) |

\* E3 = greedy full-box IoU (associate_by_iou); no pose keypoints.

**Cell notation:** `switch_rate% (IDSW / total_condition_frames)`. Denominators match Table 1 exactly.
- For example, under *Very High Overlap*, E0 has **10 switches across 252 condition frames** (**4.0%** switch rate).
- Under *All Eligible*, E0 has **155 switches across 20,988 frames** (**0.7%** overall switch rate).
- **E0 vs. E8 Very High Overlap equality (10 switches each):** Both independently calculate to exactly 10 switches, but they are distinct events: only 4 switches coincide on the same video/frame, 6 occur only in E0 (where E8 tracks smoothly), and 6 occur only in E8 (where E0 tracks smoothly), which is why their association success rates differ significantly: E0 = 76.2% (192/252) vs. E8 = 89.3% (225/252).
- **Video-level distribution (cases with switches / total cases with condition):**
  - E0 (ByteTrack): Very High 8/87 vids | High 24/315 vids | Low 71/616 vids | High Spd 32/521 vids | Low Spd 80/622 vids | Front 27/452 vids | Behind 22/463 vids | All 95/625 vids
  - E2 (Torso ROI): Very High 12/87 vids | High 36/315 vids | Low 75/616 vids | High Spd 28/521 vids | Low Spd 93/622 vids | Front 33/452 vids | Behind 37/463 vids | All 106/625 vids
  - E3 (Pose Align*): Very High 11/87 vids | High 28/315 vids | Low 65/616 vids | High Spd 25/521 vids | Low Spd 81/622 vids | Front 31/452 vids | Behind 32/463 vids | All 94/625 vids
  - E5 (Hip Colour): Very High 21/87 vids | High 106/315 vids | Low 320/616 vids | High Spd 129/521 vids | Low Spd 327/622 vids | Front 146/452 vids | Behind 184/463 vids | All 370/625 vids
  - E7 (SAM Mask): Very High 34/87 vids | High 103/315 vids | Low 187/616 vids | High Spd 87/521 vids | Low Spd 240/622 vids | Front 114/452 vids | Behind 108/463 vids | All 265/625 vids
  - E8 (MiDaS Depth): Very High 8/87 vids | High 41/315 vids | Low 137/616 vids | High Spd 48/521 vids | Low Spd 144/622 vids | Front 70/452 vids | Behind 59/463 vids | All 165/625 vids

**Identity-switch definition & condition attribution:**
- **IDSW definition:** An ID switch is counted when the heuristic's assigned track ID on the GT raider changes between consecutive observed raider frames in a video (`current_track_id != last_track_id`). Track initialization (birth) and missing tracker predictions do not increment IDSW. If a track drops out and returns with the same ID, no switch is counted.
- **Condition attribution rule:** Each switch transition is attributed to the ground-truth condition observed at the **transition frame** (the frame where the switch takes effect / the new track ID is first observed on the raider).
- **Unclassifiable events:** Transition frames with non-applicable or unknown condition labels (`overlap = NOT_APPLICABLE` where no other player is in view, `speed = UNKNOWN`, or `depth` in `UNKNOWN` / `NOT_APPLICABLE` / `BLOCKED`) are not forced into condition columns. All valid switches are included in `All Eligible`.
- **Non-additive dimensions:** Condition columns represent distinct, partially overlapping factors (overlap, speed, depth) and do not sum to the overall count.

## C01–C19 diagnostic (selected cases) — kept for comparison, separate population

| Heuristic | Very high overlap | High overlap | Low overlap | High speed | Low speed | Raider in front | Raider behind | All eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 (ByteTrack) | 28.6% (2/7) | 66.7% (36/54) | 81.1% (129/159) | 71.0% (22/31) | 76.7% (145/189) | 66.7% (34/51) | 88.8% (95/107) | 75.9% (167/220) |
| E2 (Torso ROI) | 100.0% (7/7) | 75.9% (41/54) | 84.3% (134/159) | 67.7% (21/31) | 85.2% (161/189) | 88.2% (45/51) | 86.9% (93/107) | 82.7% (182/220) |
| E3 (Pose Align*) | 42.9% (3/7) | 66.7% (36/54) | 85.5% (136/159) | 74.2% (23/31) | 80.4% (152/189) | 82.4% (42/51) | 87.9% (94/107) | 79.5% (175/220) |
| E5 (Hip Colour) | 42.9% (3/7) | 50.0% (27/54) | 73.0% (116/159) | 74.2% (23/31) | 65.1% (123/189) | 76.5% (39/51) | 58.9% (63/107) | 66.4% (146/220) |
| E7 (SAM Mask) | 57.1% (4/7) | 57.4% (31/54) | 88.1% (140/159) | 67.7% (21/31) | 81.5% (154/189) | 90.2% (46/51) | 72.9% (78/107) | 79.5% (175/220) |
| E8 (MiDaS Depth) | 100.0% (7/7) | 87.0% (47/54) | 78.6% (125/159) | 71.0% (22/31) | 83.1% (157/189) | 92.2% (47/51) | 82.2% (88/107) | 81.4% (179/220) |

The 9 selected diagnostic cases only (227 window frames, 220 eligible). Do not pool with the full-dataset table.
