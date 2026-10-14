# Condition-Wise Heuristic Performance

This file has two parts with **different populations that must not be mixed**:

1. **Full-dataset quantitative evaluation** — the principal result: every reviewed-GT sequence in `dhruba_dataset_processed`.
2. **C01–C19 qualitative diagnostic analysis** — the earlier selected-case study (9 cases with frame-condition labels), kept for failure-mode explanation.

---

# Part 1 — Full-dataset quantitative evaluation

**Coverage:** 625 videos, 22973 frames in their GT ranges, **20988 eligible raider frames** from 625 videos. 96 further sequences were excluded (raw-tracker schema, not reviewed GT; see `classification_summary.md`). All methods were run on the same GT boxes from the first GT frame of each sequence; outputs: `condition_classification/cache/<video>.tracks.json`.

> [!WARNING]
> **E3 does not implement pose alignment**: the harness (and the standalone E3 tracker) associate by greedy full-box IoU (`associate_by_iou`); pose keypoints are never used for matching. E7 (MobileSAM mask-IoU greedy association) and E8 (MiDaS depth in a Hungarian (1−IoU)+0.01·|Δdepth| cost) use the implementations fixed on 2026-10-13 and were run on all videos in this part. Prompts for which SAM returned an empty mask are treated as empty masks (IoU 0) — see `classification_summary.md`.

## Raider identity-association success rate by frame condition (full dataset)

| Heuristic | High overlap | Low overlap | High speed | Low speed | Raider in front | Raider behind | All eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 (ByteTrack) | 93.2% (2203/2365) | 95.1% (15646/16446) | 92.9% (2916/3138) | 94.9% (16865/17776) | 95.6% (5643/5905) | 96.3% (5716/5933) | 94.6% (19849/20988) |
| E2 (Torso ROI) | 90.1% (2132/2365) | 93.6% (15401/16446) | 94.2% (2955/3138) | 93.0% (16538/17776) | 92.9% (5483/5905) | 94.3% (5597/5933) | 93.2% (19556/20988) |
| E3 (Pose Align*) | 91.5% (2165/2365) | 94.1% (15483/16446) | 94.7% (2971/3138) | 93.6% (16639/17776) | 93.1% (5499/5905) | 94.9% (5633/5933) | 93.7% (19674/20988) |
| E5 (Hip Colour) | 70.0% (1656/2365) | 72.9% (11992/16446) | 72.3% (2270/3138) | 72.9% (12960/17776) | 71.2% (4204/5905) | 73.6% (4364/5933) | 72.8% (15274/20988) |
| E7 (SAM Mask) | 69.0% (1633/2365) | 85.6% (14070/16446) | 78.7% (2471/3138) | 84.3% (14981/17776) | 83.4% (4924/5905) | 84.1% (4987/5933) | 83.4% (17507/20988) |
| E8 (MiDaS Depth) | 88.4% (2091/2365) | 90.5% (14885/16446) | 90.2% (2832/3138) | 90.5% (16093/17776) | 87.4% (5162/5905) | 92.5% (5487/5933) | 90.5% (18991/20988) |

\* E3 = greedy full-box IoU (associate_by_iou); no pose keypoints.

Success rate = correctly associated eligible raider frames / eligible raider frames in that condition × 100, pooled over all frames of all videos (the All-eligible column is computed on the full eligible population, not averaged over conditions). Numerators/denominators and Wilson intervals: `summary_cell_counts.csv`; per-video counts: `summary_cell_counts_by_case.csv`; per-frame verdicts: `frame_heuristic_association.csv`.

### Success definition

- Per video × heuristic, the **matched ID** is the heuristic track ID found on the GT raider box in the most eligible frames (IDF1-style identity match for one GT trajectory).
- A frame is correct iff the GT raider box carries the matched ID and no other box in that frame does. ID switches, re-born tracks, missing tracker outputs, and the raider taking another player's ID are failures.
- Eligible = GT record present, original JPG present, exactly one raider (or a duplicate resolved by continuity). Ineligible frames are excluded from every denominator; UNKNOWN / NOT_APPLICABLE depth frames are excluded from the two depth columns only.

## Supporting information

| Condition | Eligible frames | Videos contributing |
|---|---:|---:|
| High overlap | 2365 | 315 |
| Low overlap | 16446 | 616 |
| High speed | 3138 | 521 |
| Low speed | 17776 | 622 |
| Raider in front | 5905 | 452 |
| Raider behind | 5933 | 463 |
| All eligible | 20988 | 625 |

- Ineligible frames: 1985 (NO_RAIDER_GT_BOX: 1881, AMBIGUOUS_DUPLICATE_RAIDER_GT: 12, NO_GT_RECORD: 92).
- Overlap labels (eligible frames): VERY HIGH 252, HIGH 2365, LOW 16446, NOT_APPLICABLE (no other player) 1925. Speed labels: HIGH 3138, LOW 17776, UNKNOWN 74. Frames with a non-applicable/unknown label are excluded only from that factor's columns.
- Depth labels (eligible frames): FRONT 5905, BEHIND 5933, UNKNOWN 4299, NOT_APPLICABLE 4851, BLOCKED 0. Depth computed with the existing MiDaS_small implementation for every frame that needed it.
- Thresholds (unchanged): speed HIGH if > 1050 px/s; depth FRONT/BEHIND if |normalised difference| > 0.05.
- Output mapping check: every heuristic read from its own column of the per-video tracking output; E3 vs E7 outputs differ on 6227 frames, E0 vs E8 on 12200 frames. Raider box matched in every heuristic output for every eligible frame: True. Videos missing a cache: 0.
- **2026-10-14 correction:** the prior serializer substituted a GT ID whenever a heuristic had no assignment. The current outputs retain each method's actual new ID or missing value; missing values are failures. The old outputs and summaries are preserved in `diagnostic_analysis/backups/pre_gt_id_fallback_fix_2026-10-14/`; 5 incidental first-frame GT-ID signatures remain.

## Difference to E0 with video-level bootstrap

Heuristic − E0 success rate on the same frames, in percentage points; 95% interval from 2000 bootstrap resamples of **videos** (keeps within-video correlation). † = interval excludes 0.

| Heuristic − E0 (pp) | High overlap | Low overlap | High speed | Low speed | Raider in front | Raider behind | All eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2 | -3.0 [-6.1, -0.1] † | -1.5 [-3.3, +0.2] | +1.2 [-0.5, +3.0] | -1.8 [-3.8, -0.1] † | -2.7 [-5.3, +0.0] | -2.0 [-4.0, -0.2] † | -1.4 [-3.2, +0.2] |
| E3 | -1.6 [-4.7, +1.4] | -1.0 [-2.9, +0.7] | +1.8 [-0.2, +3.6] | -1.3 [-3.2, +0.5] | -2.4 [-5.1, +0.2] | -1.4 [-3.2, +0.2] | -0.8 [-2.5, +0.8] |
| E5 | -23.1 [-28.4, -18.5] † | -22.2 [-24.9, -19.7] † | -20.6 [-23.9, -17.2] † | -22.0 [-24.7, -19.5] † | -24.4 [-28.6, -20.1] † | -22.8 [-26.1, -19.7] † | -21.8 [-24.3, -19.4] † |
| E7 | -24.1 [-29.6, -19.1] † | -9.6 [-11.8, -7.4] † | -14.2 [-16.9, -11.6] † | -10.6 [-12.9, -8.4] † | -12.2 [-15.6, -9.0] † | -12.3 [-15.3, -9.6] † | -11.2 [-13.3, -9.0] † |
| E8 | -4.7 [-8.5, -1.2] † | -4.6 [-6.7, -2.7] † | -2.7 [-5.2, -0.1] † | -4.3 [-6.4, -2.5] † | -8.2 [-11.2, -5.1] † | -3.9 [-5.9, -2.0] † | -4.1 [-6.1, -2.2] † |

## Per-video consistency

Number of videos where the heuristic's per-video success rate is higher / equal / lower than E0's (videos with ≥1 frame in that condition).

| Heuristic vs E0 | High overlap | Low overlap | High speed | Low speed | Raider in front | Raider behind | All eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2 | 24 / 262 / 29 | 79 / 483 / 54 | 43 / 450 / 28 | 92 / 469 / 61 | 34 / 382 / 36 | 40 / 392 / 31 | 98 / 465 / 62 |
| E3 | 28 / 265 / 22 | 81 / 491 / 44 | 44 / 453 / 24 | 97 / 475 / 50 | 34 / 388 / 30 | 41 / 396 / 26 | 103 / 473 / 49 |
| E5 | 20 / 160 / 135 | 43 / 182 / 391 | 28 / 255 / 238 | 47 / 169 / 406 | 21 / 190 / 241 | 25 / 200 / 238 | 49 / 159 / 417 |
| E7 | 14 / 182 / 119 | 52 / 367 / 197 | 23 / 341 / 157 | 58 / 339 / 225 | 19 / 299 / 134 | 27 / 300 / 136 | 59 / 334 / 232 |
| E8 | 22 / 248 / 45 | 76 / 424 / 116 | 44 / 405 / 72 | 86 / 417 / 119 | 30 / 343 / 79 | 39 / 353 / 71 | 89 / 412 / 124 |

## Sensitivity: first-frame anchor

| Heuristic | High overlap | Low overlap | High speed | Low speed | Raider in front | Raider behind | All eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 88.3% (2088/2365) | 89.0% (14634/16446) | 87.7% (2751/3138) | 89.0% (15816/17776) | 91.9% (5427/5905) | 90.1% (5346/5933) | 88.8% (18638/20988) |
| E2 | 85.7% (2026/2365) | 90.0% (14800/16446) | 91.7% (2877/3138) | 89.1% (15845/17776) | 87.7% (5181/5905) | 93.0% (5516/5933) | 89.5% (18787/20988) |
| E3 | 88.1% (2084/2365) | 90.8% (14938/16446) | 93.0% (2917/3138) | 90.1% (16012/17776) | 88.7% (5237/5905) | 94.4% (5602/5933) | 90.5% (18995/20988) |
| E5 | 31.2% (739/2365) | 41.6% (6843/16446) | 38.5% (1207/3138) | 41.2% (7329/17776) | 38.7% (2286/5905) | 39.6% (2351/5933) | 40.8% (8569/20988) |
| E7 | 57.3% (1356/2365) | 78.7% (12951/16446) | 70.7% (2220/3138) | 76.5% (13601/17776) | 77.1% (4551/5905) | 77.4% (4590/5933) | 75.6% (15877/20988) |
| E8 | 82.9% (1960/2365) | 85.7% (14088/16446) | 82.6% (2592/3138) | 85.8% (15257/17776) | 82.1% (4846/5905) | 89.6% (5315/5933) | 85.4% (17915/20988) |

## Interpretation (generated from the counts above)

- **High overlap** (N = 2365 frames, 315 videos): the baseline **E0** is highest at 93.2%; best alternative E3 91.5% (-1.6 pp; video-bootstrap 95% CI of E3 − E0: [-4.7, +1.4] pp (includes 0)).
- **Low overlap** (N = 16446 frames, 616 videos): the baseline **E0** is highest at 95.1%; best alternative E3 94.1% (-1.0 pp; video-bootstrap 95% CI of E3 − E0: [-2.9, +0.7] pp (includes 0)).
- **High speed** (N = 3138 frames, 521 videos): highest observed rate **E3** 94.7% vs E0 92.9%; difference to E0 +1.8 pp, video-bootstrap 95% CI [-0.2, +3.6] (includes 0). Runner-up: E2 94.2%.
- **Low speed** (N = 17776 frames, 622 videos): the baseline **E0** is highest at 94.9%; best alternative E3 93.6% (-1.3 pp; video-bootstrap 95% CI of E3 − E0: [-3.2, +0.5] pp (includes 0)).
- **Raider in front** (N = 5905 frames, 452 videos): the baseline **E0** is highest at 95.6%; best alternative E3 93.1% (-2.4 pp; video-bootstrap 95% CI of E3 − E0: [-5.1, +0.2] pp (includes 0)).
- **Raider behind** (N = 5933 frames, 463 videos): the baseline **E0** is highest at 96.3%; best alternative E3 94.9% (-1.4 pp; video-bootstrap 95% CI of E3 − E0: [-3.2, +0.2] pp (includes 0)).
- **All eligible** (N = 20988 frames, 625 videos): the baseline **E0** is highest at 94.6%; best alternative E3 93.7% (-0.8 pp; video-bootstrap 95% CI of E3 − E0: [-2.5, +0.8] pp (includes 0)).
- **Hardest conditions across all methods** (lowest best-case rate): High overlap (best 93.2%), High speed (best 94.7%).
- E0: low−high overlap +2.0 pp; low−high speed +1.9 pp; front−behind -0.8 pp.
- E2: low−high overlap +3.5 pp; low−high speed -1.1 pp; front−behind -1.5 pp.
- E3: low−high overlap +2.6 pp; low−high speed -1.1 pp; front−behind -1.8 pp.
- E5: low−high overlap +2.9 pp; low−high speed +0.6 pp; front−behind -2.4 pp.
- E7: low−high overlap +16.5 pp; low−high speed +5.5 pp; front−behind -0.7 pp.
- E8: low−high overlap +2.1 pp; low−high speed +0.3 pp; front−behind -5.1 pp.
- **Differences to E0 whose video-level bootstrap interval excludes 0:** E2 High overlap -3.0 pp; E2 Low speed -1.8 pp; E2 Raider behind -2.0 pp; E5 High overlap -23.1 pp; E5 Low overlap -22.2 pp; E5 High speed -20.6 pp; E5 Low speed -22.0 pp; E5 Raider in front -24.4 pp; E5 Raider behind -22.8 pp; E5 All eligible -21.8 pp; E7 High overlap -24.1 pp; E7 Low overlap -9.6 pp; E7 High speed -14.2 pp; E7 Low speed -10.6 pp; E7 Raider in front -12.2 pp; E7 Raider behind -12.3 pp; E7 All eligible -11.2 pp; E8 High overlap -4.7 pp; E8 Low overlap -4.6 pp; E8 High speed -2.7 pp; E8 Low speed -4.3 pp; E8 Raider in front -8.2 pp; E8 Raider behind -3.9 pp; E8 All eligible -4.1 pp. This is an uncertainty estimate that respects within-video correlation, not a formal significance test with multiple-comparison control (36 comparisons).
- **Cue caveats:** E3 never uses pose keypoints (greedy full-box IoU), so no conclusion about pose alignment is possible. E7 vs E3 isolates the SAM-mask cue (same greedy procedure and 0.15 threshold). E8 vs E0 is not a clean depth ablation: E8 is frame-to-frame Hungarian matching with a depth term and no Kalman motion model / track buffer.
- **Limitations:** consecutive frames are strongly correlated (one switch changes many frame verdicts); videos differ in length (frames per video vary widely), so long videos weigh more in pooled rates; condition frequencies are uneven (see coverage table); depth labels exist only where an overlapping or nearby opponent exists and the depth difference exceeds the margin.

---

# Part 2 — C01–C19 qualitative diagnostic analysis (selected cases)

Separate population: the 9 selected diagnostic cases (C01, C03, C04, C05, C06, C07, C10, C14, C19) with their diagnostic frame windows (`condition_classification/C*/frame_conditions.csv`, `diagnostic_analysis/C*/tracks_E*.json`). Use it to explain specific successes and failures together with `OBSERVATIONS.md`, `CASE_MATRIX.md` and `HEURISTIC_SUMMARY.md`; do not pool it with Part 1.

| Heuristic | High overlap | Low overlap | High speed | Low speed | Raider in front | Raider behind | All eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 (ByteTrack) | 66.7% (36/54) | 81.1% (129/159) | 71.0% (22/31) | 76.7% (145/189) | 66.7% (34/51) | 88.8% (95/107) | 75.9% (167/220) |
| E2 (Torso ROI) | 75.9% (41/54) | 84.3% (134/159) | 67.7% (21/31) | 85.2% (161/189) | 88.2% (45/51) | 86.9% (93/107) | 82.7% (182/220) |
| E3 (Pose Align*) | 66.7% (36/54) | 85.5% (136/159) | 74.2% (23/31) | 80.4% (152/189) | 82.4% (42/51) | 87.9% (94/107) | 79.5% (175/220) |
| E5 (Hip Colour) | 50.0% (27/54) | 73.0% (116/159) | 74.2% (23/31) | 65.1% (123/189) | 76.5% (39/51) | 58.9% (63/107) | 66.4% (146/220) |
| E7 (SAM Mask) | 57.4% (31/54) | 88.1% (140/159) | 67.7% (21/31) | 81.5% (154/189) | 90.2% (46/51) | 72.9% (78/107) | 79.5% (175/220) |
| E8 (MiDaS Depth) | 87.0% (47/54) | 78.6% (125/159) | 71.0% (22/31) | 83.1% (157/189) | 92.2% (47/51) | 82.2% (88/107) | 81.4% (179/220) |

Diagnostic switch counts recomputed from the case files match `experiment_results_summary.json`: True.
