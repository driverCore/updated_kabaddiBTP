"""
full_dataset_reports.py
=======================
Markdown rendering for run_full_dataset_evaluation.py (evaluate stage). Writes the full-dataset
results into the EXISTING report files (condition_classification/HEURISTIC_CONDITION_SUMMARY.md,
condition_classification/classification_summary.md, and a marked section of diagnostic_analysis/README.md),
keeping the C01–C19 diagnostic results as a separately labelled section.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

VERY_HIGH_IOU = 0.5
README_START = "<!-- FULL_DATASET_RESULTS_START -->"
README_END = "<!-- FULL_DATASET_RESULTS_END -->"


def _f(c) -> str:
    return "n/a" if c["pct"] is None else f"{c['pct']:.1f}% ({c['k']}/{c['n']})"


def _very_high(assoc, rows, key_case="case_id") -> Dict[str, Dict[str, Any]]:
    iou = {(str(r["case_id"]), str(r["frame_number"])): r["max_overlap_iou"] for r in rows}
    out = {}
    for h in sorted({a["heuristic"] for a in assoc}):
        sel = [a for a in assoc if a["heuristic"] == h
               and iou.get((str(a["case_id"]), str(a["frame_number"])), "") not in ("", None)
               and float(iou[(str(a["case_id"]), str(a["frame_number"]))]) > VERY_HIGH_IOU]
        k = sum(a["correct"] for a in sel)
        out[h] = {"k": k, "n": len(sel), "pct": 100 * k / len(sel) if sel else None,
                  "cases": len({a["case_id"] for a in sel})}
    return out


def _success_table(shc, table, ident, vh=None) -> List[str]:
    labels = [c[2] for c in shc.CONDITIONS]
    head = ["Heuristic"] + (["Very high overlap"] if vh else []) + labels + ["All eligible"]
    L = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for h in shc.HEURISTICS:
        star = "*" if h in shc.starred(ident) else ""
        cells = ([_f(vh[h])] if vh else []) + [_f(table[h][l]) for l in labels + ["All eligible"]]
        L.append(f"| {h} ({shc.HEURISTIC_LABELS[h]}{star}) | " + " | ".join(cells) + " |")
    return L


def _coverage(ctx) -> Dict[str, Any]:
    rows = ctx["rows"]
    el = [r for r in rows if r["eligible"] == 1]
    return {"videos": len({r["case_id"] for r in rows}), "frames": len(rows), "eligible": len(el),
            "eligible_videos": len({r["case_id"] for r in el}),
            "inel": Counter(r["ineligible_reason"] for r in rows if r["eligible"] != 1)}


def _per_video_consistency(ctx) -> List[str]:
    shc, by = ctx["shc"], ctx["by_video"]
    rate = defaultdict(dict)
    for r in by:
        rate[(r["column_label"], r["case_id"])][r["heuristic"]] = (r["success_rate_pct"], r["n_eligible_frames"])
    labels = [c[2] for c in shc.CONDITIONS] + ["All eligible"]
    L = ["| Heuristic vs E0 | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for h in [x for x in shc.HEURISTICS if x != "E0"]:
        cells = []
        for lab in labels:
            better = worse = equal = 0
            for (l, vid), d in rate.items():
                if l != lab or h not in d or "E0" not in d:
                    continue
                if d[h][0] > d["E0"][0]:
                    better += 1
                elif d[h][0] < d["E0"][0]:
                    worse += 1
                else:
                    equal += 1
            cells.append(f"{better} / {equal} / {worse}")
        L.append(f"| {h} | " + " | ".join(cells) + " |")
    return L


def _boot_table(ctx) -> List[str]:
    shc, boot = ctx["shc"], ctx["boot"]
    labels = [c[2] for c in shc.CONDITIONS] + ["All eligible"]
    L = ["| Heuristic − E0 (pp) | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for h, d in boot.items():
        cells = []
        for lab in labels:
            b = d.get(lab)
            if b is None:
                cells.append("n/a")
            else:
                sig = "" if b[1] <= 0 <= b[2] else " †"
                cells.append(f"{b[0]:+.1f} [{b[1]:+.1f}, {b[2]:+.1f}]{sig}")
        L.append(f"| {h} | " + " | ".join(cells) + " |")
    return L


def _interpretation(ctx) -> List[str]:
    shc, table, boot, same = ctx["shc"], ctx["table"], ctx["boot"], ctx["same"]
    labels = [c[2] for c in shc.CONDITIONS]
    L = ["## Interpretation (generated from the counts above)\n"]
    for lab in labels + ["All eligible"]:
        vals = sorted(((table[h][lab]["pct"], h) for h in shc.HEURISTICS if table[h][lab]["pct"] is not None),
                      reverse=True)
        top, h = vals[0]
        e0 = table["E0"][lab]["pct"]
        b = boot.get(h, {}).get(lab) if h != "E0" else None
        btxt = (f"; difference to E0 {b[0]:+.1f} pp, video-bootstrap 95% CI [{b[1]:+.1f}, {b[2]:+.1f}]"
                f"{' (excludes 0)' if not (b[1] <= 0 <= b[2]) else ' (includes 0)'}") if b else ""
        if h == "E0":
            alt_v, alt = vals[1]
            ab = boot.get(alt, {}).get(lab)
            atxt = (f"; video-bootstrap 95% CI of {alt} − E0: [{ab[1]:+.1f}, {ab[2]:+.1f}] pp"
                    f"{' (excludes 0)' if not (ab[1] <= 0 <= ab[2]) else ' (includes 0)'}") if ab else ""
            L.append(f"- **{lab}** (N = {table[h][lab]['n']} frames, {table[h][lab]['cases']} videos): the baseline "
                     f"**E0** is highest at {top:.1f}%; best alternative {alt} {alt_v:.1f}% "
                     f"({alt_v - top:+.1f} pp{atxt}).")
        else:
            L.append(f"- **{lab}** (N = {table[h][lab]['n']} frames, {table[h][lab]['cases']} videos): highest observed "
                     f"rate **{h}** {top:.1f}% vs E0 {e0:.1f}%{btxt}. Runner-up: {vals[1][1]} {vals[1][0]:.1f}%.")
    best = {lab: max(table[h][lab]["pct"] for h in shc.HEURISTICS) for lab in labels}
    hard = sorted(best.items(), key=lambda x: x[1])
    L.append(f"- **Hardest conditions across all methods** (lowest best-case rate): {hard[0][0]} "
             f"(best {hard[0][1]:.1f}%), {hard[1][0]} (best {hard[1][1]:.1f}%).")
    for h in shc.HEURISTICS:
        t = table[h]
        L.append(f"- {h}: low−high overlap {t['Low overlap']['pct'] - t['High overlap']['pct']:+.1f} pp; "
                 f"low−high speed {t['Low speed']['pct'] - t['High speed']['pct']:+.1f} pp; "
                 f"front−behind {t['Raider in front']['pct'] - t['Raider behind']['pct']:+.1f} pp.")
    sig = [(h, lab, b) for h, d in boot.items() for lab, b in d.items() if b and not (b[1] <= 0 <= b[2])]
    if sig:
        L.append("- **Differences to E0 whose video-level bootstrap interval excludes 0:** "
                 + "; ".join(f"{h} {lab} {b[0]:+.1f} pp" for h, lab, b in sig)
                 + ". This is an uncertainty estimate that respects within-video correlation, not a formal "
                   "significance test with multiple-comparison control (36 comparisons).")
    else:
        L.append("- No heuristic − E0 difference has a video-level bootstrap interval that excludes 0.")
    L.append("- **Cue caveats:** E3 never uses pose keypoints (greedy full-box IoU), so no conclusion about pose "
             "alignment is possible. E7 vs E3 isolates the SAM-mask cue (same greedy procedure and 0.15 threshold). "
             "E8 vs E0 is not a clean depth ablation: E8 is frame-to-frame Hungarian matching with a depth term and no "
             "Kalman motion model / track buffer.")
    L.append("- **Limitations:** consecutive frames are strongly correlated (one switch changes many frame verdicts); "
             "videos differ in length (frames per video vary widely), so long videos weigh more in pooled rates; "
             "condition frequencies are uneven (see coverage table); depth labels exist only where an overlapping or "
             "nearby opponent exists and the depth difference exceeds the margin.")
    return L


def heuristic_md(ctx) -> str:
    shc, table, val, same = ctx["shc"], ctx["table"], ctx["validation"], ctx["same"]
    cov = _coverage(ctx)
    labels = [c[2] for c in shc.CONDITIONS]
    L: List[str] = ["# Condition-Wise Heuristic Performance\n"]
    L.append("This file has two parts with **different populations that must not be mixed**:\n")
    L.append("1. **Full-dataset quantitative evaluation** — the principal result: every reviewed-GT sequence in "
             "`dhruba_dataset_processed`.")
    L.append("2. **C01–C19 qualitative diagnostic analysis** — the earlier selected-case study (9 cases with frame-"
             "condition labels), kept for failure-mode explanation.\n")
    L.append("---\n\n# Part 1 — Full-dataset quantitative evaluation\n")
    L.append(f"**Coverage:** {cov['videos']} videos, {cov['frames']} frames in their GT ranges, "
             f"**{cov['eligible']} eligible raider frames** from {cov['eligible_videos']} videos. "
             f"{len(ctx['excluded'])} further sequences were excluded (raw-tracker schema, not reviewed GT; see "
             f"`classification_summary.md`). All methods were run on the same GT boxes from the first GT frame of each "
             f"sequence; outputs: `condition_classification/cache/<video>.tracks.json`.\n")
    L.append("> [!WARNING]\n> **E3 does not implement pose alignment**: the harness (and the standalone E3 tracker) "
             "associate by greedy full-box IoU (`associate_by_iou`); pose keypoints are never used for matching. "
             "E7 (MobileSAM mask-IoU greedy association) and E8 (MiDaS depth in a Hungarian (1−IoU)+0.01·|Δdepth| cost) "
             "use the implementations fixed on 2026-10-13 and were run on all videos in this part. Prompts for which "
             "SAM returned an empty mask are treated as empty masks (IoU 0) — see `classification_summary.md`.\n")
    L.append("## Raider identity-association success rate by frame condition (full dataset)\n")
    L += _success_table(shc, table, same)
    L.append("\n\\* " + "; ".join(f"{h} = {shc.implemented_as(h, same)}" for h in shc.starred(same)) + ".\n")
    L.append("Success rate = correctly associated eligible raider frames / eligible raider frames in that condition × 100, "
             "pooled over all frames of all videos (the All-eligible column is computed on the full eligible population, "
             "not averaged over conditions). Numerators/denominators and Wilson intervals: `summary_cell_counts.csv`; "
             "per-video counts: `summary_cell_counts_by_case.csv`; per-frame verdicts: `frame_heuristic_association.csv`.\n")
    L.append("### Success definition\n")
    L.append("- Per video × heuristic, the **matched ID** is the heuristic track ID found on the GT raider box in the most "
             "eligible frames (IDF1-style identity match for one GT trajectory).")
    L.append("- A frame is correct iff the GT raider box carries the matched ID and no other box in that frame does. ID "
             "switches, re-born tracks, missing tracker outputs, and the raider taking another player's ID are failures.")
    L.append("- Eligible = GT record present, original JPG present, exactly one raider (or a duplicate resolved by "
             "continuity). Ineligible frames are excluded from every denominator; UNKNOWN / NOT_APPLICABLE depth frames "
             "are excluded from the two depth columns only.\n")

    L.append("## Supporting information\n")
    L.append("| Condition | Eligible frames | Videos contributing |\n|---|---:|---:|")
    for lab in labels + ["All eligible"]:
        L.append(f"| {lab} | {table['E0'][lab]['n']} | {table['E0'][lab]['cases']} |")
    dc = Counter(r["depth_label"] for r in ctx["rows"] if r["eligible"] == 1)
    L.append("")
    L.append(f"- Ineligible frames: {sum(cov['inel'].values())} ({', '.join(f'{k}: {v}' for k, v in cov['inel'].items())}).")
    oc = Counter(r["overlap_label"] for r in ctx["rows"] if r["eligible"] == 1)
    sc = Counter(r["speed_label"] for r in ctx["rows"] if r["eligible"] == 1)
    L.append(f"- Overlap labels (eligible frames): VERY HIGH {oc['VERY_HIGH_OVERLAP']}, HIGH {oc['HIGH_OVERLAP']}, LOW {oc['LOW_OVERLAP']}, NOT_APPLICABLE "
             f"(no other player) {oc['NOT_APPLICABLE']}. Speed labels: HIGH {sc['HIGH_SPEED']}, LOW {sc['LOW_SPEED']}, "
             f"UNKNOWN {sc['UNKNOWN']}. Frames with a non-applicable/unknown label are excluded only from that factor's columns.")
    L.append(f"- Depth labels (eligible frames): FRONT {dc['FRONT']}, BEHIND {dc['BEHIND']}, UNKNOWN {dc['UNKNOWN']}, "
             f"NOT_APPLICABLE {dc['NOT_APPLICABLE']}, BLOCKED {dc['BLOCKED']}. Depth computed with the existing MiDaS_small "
             f"implementation for every frame that needed it.")
    L.append("- Thresholds (unchanged): speed HIGH if > 1050 px/s; depth FRONT/BEHIND if |normalised difference| > 0.05.")
    L.append(f"- Output mapping check: every heuristic read from its own column of the per-video tracking output; "
             f"E3 vs E7 outputs differ on {val['frames_where_outputs_differ']['E3==E7']} frames, E0 vs E8 on "
             f"{val['frames_where_outputs_differ']['E0==E8']} frames. Raider box matched in every heuristic output for every "
             f"eligible frame: {not val['box_match_failures']}. Videos missing a cache: {len(val['videos_missing_cache'])}.")
    L.append("- **2026-10-14 correction:** the prior serializer substituted a GT ID whenever a heuristic had no assignment. "
             "The current outputs retain each method's actual new ID or missing value; missing values are failures. "
             f"The old outputs and summaries are preserved in `diagnostic_analysis/backups/pre_gt_id_fallback_fix_2026-10-14/`; "
             f"{val.get('birth_placeholder_suspects_count', 0)} incidental first-frame GT-ID signatures remain.\n")

    L.append("## Difference to E0 with video-level bootstrap\n")
    L.append("Heuristic − E0 success rate on the same frames, in percentage points; 95% interval from 2000 bootstrap "
             "resamples of **videos** (keeps within-video correlation). † = interval excludes 0.\n")
    L += _boot_table(ctx)
    L.append("\n## Per-video consistency\n")
    L.append("Number of videos where the heuristic's per-video success rate is higher / equal / lower than E0's "
             "(videos with ≥1 frame in that condition).\n")
    L += _per_video_consistency(ctx)
    L.append("\n## Sensitivity: first-frame anchor\n")
    L.append("| Heuristic | " + " | ".join(labels) + " | All eligible |")
    L.append("|---|" + "---:|" * (len(labels) + 1))
    for h in shc.HEURISTICS:
        cells = []
        for field, value, lab in shc.CONDITIONS + [shc.ALL]:
            c = shc.cell(ctx["assoc"], h, field, value, "correct_first_frame_anchor")
            cells.append(_f(c))
        L.append(f"| {h} | " + " | ".join(cells) + " |")
    L.append("")
    L += _interpretation(ctx)

    L.append("\n---\n\n# Part 2 — C01–C19 qualitative diagnostic analysis (selected cases)\n")
    L.append("Separate population: the 9 selected diagnostic cases (C01, C03, C04, C05, C06, C07, C10, C14, C19) with "
             "their diagnostic frame windows (`condition_classification/C*/frame_conditions.csv`, "
             "`diagnostic_analysis/C*/tracks_E*.json`). Use it to explain specific successes and failures together with "
             "`OBSERVATIONS.md`, `CASE_MATRIX.md` and `HEURISTIC_SUMMARY.md`; do not pool it with Part 1.\n")
    L += _success_table(shc, ctx["d_table"], ctx["d_ident"])
    L.append(f"\nDiagnostic switch counts recomputed from the case files match `experiment_results_summary.json`: "
             f"{ctx['d_val']['all_switch_counts_match']}.")
    return "\n".join(L) + "\n"


def classification_md(ctx) -> str:
    rows, cfg, shc = ctx["rows"], ctx["cfg"], ctx["shc"]
    cov = _coverage(ctx)
    action = {v["video_id"]: f"{v['split']}/{v['action']}" for v in ctx["vids"]}
    L: List[str] = ["# Frame-Condition Classification\n"]
    L.append("Principal population: **full dataset** — every sequence in `dhruba_dataset_processed` with reviewed GT "
             "tracks, each over its full GT frame range. The earlier selected-case (C01–C19 diagnostic) results are kept "
             "at the end of this file and in `condition_classification/C*/`.\n")
    L.append(f"**Depth factor status:** computed with the existing MiDaS_small implementation "
             f"(`tracking/E8_midas_depth/depth_features.py`).\n")
    L.append("## Dataset coverage\n")
    allv = ctx["all_vids"]
    L.append(f"- Sequences with a GT `tracks.json`: {len(allv)}. **Included (reviewed GT schema): {len(ctx['vids'])}.** "
             f"Excluded: {len(ctx['excluded'])} sequences in the raw-tracker schema (records without `frame_idx`, boxes "
             f"without `status`/`original_id`, with `bbox_yolo` and detector confidences): never opened in the GT editor, "
             f"so they are not ground truth. Excluded by class: "
             + ", ".join(f"{k}: {v}" for k, v in sorted(Counter('/'.join(x.split('/')[:2]) for x in ctx['excluded']).items())) + ".")
    L.append(f"- Frames in the included GT ranges: {cov['frames']}; **eligible raider frames: {cov['eligible']}** "
             f"({cov['eligible_videos']} videos). Ineligible: "
             + ", ".join(f"{k}: {v}" for k, v in cov["inel"].items()) + ".\n")

    L.append("## Frames per action class\n")
    L.append("| Class | Videos | Frames | Eligible | Very high overlap | HIGH_OVERLAP | LOW_OVERLAP | HIGH_SPEED | "
             "LOW_SPEED | FRONT | BEHIND | Depth UNKNOWN | Depth N/A (reason) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    tot = Counter()
    for cls in sorted(set(action.values())):
        cr = [r for r in rows if action.get(r["case_id"]) == cls]
        el = [r for r in cr if r["eligible"] == 1]
        o, s, d = Counter(r["overlap_label"] for r in el), Counter(r["speed_label"] for r in el), Counter(r["depth_label"] for r in el)
        vh = sum(1 for r in el if r["max_overlap_iou"] != "" and float(r["max_overlap_iou"]) > VERY_HIGH_IOU)
        na = [r for r in el if r["depth_label"] == "NOT_APPLICABLE"]
        na_none = sum(1 for r in na if r["n_comparison_players"] in (0, ""))
        vals = dict(v=len({r['case_id'] for r in cr}), f=len(cr), e=len(el), vh=vh, ho=o["HIGH_OVERLAP"], lo=o["LOW_OVERLAP"],
                    hs=s["HIGH_SPEED"], ls=s["LOW_SPEED"], su=s["UNKNOWN"], fr=d["FRONT"], be=d["BEHIND"],
                    du=d["UNKNOWN"] + d["BLOCKED"], dn=len(na), nn=na_none)
        tot.update(vals)
        L.append(f"| {cls} | {vals['v']} | {vals['f']} | {vals['e']} | {vh} | {vals['ho']} | {vals['lo']} | {vals['hs']} | "
                 f"{vals['ls']} | {vals['fr']} | {vals['be']} | {vals['du']} | "
                 f"{vals['dn']} ({vals['dn'] - na_none} no opponent within 1 raider-height; {na_none} no other player) |")
    L.append(f"| **Total** | **{tot['v']}** | **{tot['f']}** | **{tot['e']}** | **{tot['vh']}** | **{tot['ho']}** | **{tot['lo']}** | "
             f"**{tot['hs']}** | **{tot['ls']}** | **{tot['fr']}** | **{tot['be']}** | **{tot['du']}** | "
             f"**{tot['dn']}** ({tot['dn'] - tot['nn']} / {tot['nn']}) |\n")
    n_ov_na = tot["e"] - tot["ho"] - tot["lo"]
    L.append(f"Overlap is NOT_APPLICABLE for {n_ov_na} eligible frames with no other GT player in the frame; they are "
             f"excluded from both overlap columns (HIGH + LOW + N/A = Eligible).\n")
    L.append(f"Depth N/A = no eligible comparison opponent: no GT box overlaps the raider and the nearest player's centre "
             f"is more than {cfg['depth']['proximity_max_raider_heights']} raider box-height away (or no other player). "
             f"Speed UNKNOWN frames (not shown as a column): {tot['su']} — no adjacent raider observation, or the video's "
             f"JPG count differs from its frame count (FPS not trusted). Per-video counts: `summary_cell_counts_by_case.csv`; per-frame: "
             f"`all_cases_frame_conditions.csv`.\n")

    L.append("## Reliability flags (eligible frames)\n\n| Flag | Frames |\n|---|---:|")
    fc = Counter()
    for r in rows:
        for f in filter(None, str(r["flags"]).split(";")):
            fc[re.sub(r"\(.*\)", "", f)] += 1
    L += [f"| `{k}` | {v} |" for k, v in sorted(fc.items())]
    L.append("")

    L.append("## Thresholds and rationale (unchanged from the validated experiment)\n")
    L.append("- **Overlap:** LOW, HIGH, and VERY HIGH are mutually exclusive conditions.")
    L.append(f"- **Speed:** HIGH_SPEED if speed > **{cfg['speed']['high_speed_px_per_s_threshold']} px/s**. {cfg['speed']['source']} "
             f"Δt = 1/FPS of the source video; velocity only between consecutive GT raider observations.")
    L.append(f"- **Depth:** FRONT/BEHIND if |normalised median-depth difference| > **{cfg['depth']['front_behind_margin']}**. "
             f"{cfg['depth']['margin_source']}")
    L.append("- **Very high overlap:** separate band in the overlap partition, reported in the tables only.\n")

    pre = ctx["prechecks"]
    post = ctx["post"]
    L.append("## Validation\n")
    fr_bad = [p["case_id"] for p in pre if not p.get("full_rate_extraction")]
    mism = sum(len(p.get("frame_correspondence", {}).get("mismatch_frames", [])) for p in pre)
    checked = sum(p.get("frame_correspondence", {}).get("checked", 0) for p in pre)
    dup = [p["case_id"] for p in pre if p.get("duplicate_frame_numbers")]
    idx_bad = [p["case_id"] for p in pre if not p.get("frame_idx_consistent")]
    L.append(f"- JPG ↔ video frame correspondence: {checked - mism}/{checked} JPGs closest to their own decoded video frame; "
             f"{mism} flagged.")
    L.append(f"- Videos whose JPG count ≠ video frame count (speed set UNKNOWN): {len(fr_bad)}"
             + (f" ({', '.join(fr_bad[:15])}{' …' if len(fr_bad) > 15 else ''})" if fr_bad else "") + ".")
    L.append(f"- Videos with duplicate frame numbers: {len(dup)}; with frame_idx ≠ frame−1: {len(idx_bad)}.")
    for k, v in post.items():
        L.append(f"- `{k}`: {v}")
    L.append("")
    L.append("## Limitations\n")
    L.append("- Overlap is 2D box IoU, not physical occlusion. Speed is image-plane px/s (camera motion, resolution and "
             "perspective affect it). Depth is monocular relative inverse depth; the 0.05 margin is a proposed value. "
             "Many GT boxes (mostly defenders) are `unreviewed`. Consecutive frames are correlated.\n")

    # end tables
    vh = _very_high(ctx["assoc"], rows)
    L.append("## Raider identity-association success rate by condition — full dataset\n")
    L.append("Copied from `HEURISTIC_CONDITION_SUMMARY.md` (Part 1). Matched-ID (IDF1-style) success; counts in brackets.\n")
    L.append("**Raider ID association accuracy** = correctly associated eligible raider frames / eligible raider frames in the condition × 100. For each video and heuristic, the matched ID is the ID appearing on the GT raider box most often; a frame is correct only when that ID is on the raider box and on no other box. Missing IDs, switched/re-born IDs, and IDs shared with another player are incorrect.\n")
    L += _success_table(shc, ctx["table"], ctx["same"], vh)
    L.append("\n\\* " + "; ".join(f"{h} = {shc.implemented_as(h, ctx['same'])}" for h in shc.starred(ctx["same"])) + ".\n")
    L.append(f"Very high overlap is a separate overlap band ({vh['E0']['n']} frames from {vh['E0']['cases']} videos).\n")
    dvh = _very_high(ctx["assoc_diag"], ctx["d_rows"]) if "assoc_diag" in ctx else None
    L.append("## C01–C19 diagnostic (selected cases) — kept for comparison, separate population\n")
    L += _success_table(shc, ctx["d_table"], ctx["d_ident"], dvh)
    L.append("\nThe 9 selected diagnostic cases only (227 window frames, 220 eligible). Do not pool with the full-dataset "
             "table.")
    return "\n".join(L) + "\n"


def update_readme(ctx, path) -> None:
    shc = ctx["shc"]
    cov = _coverage(ctx)
    block = [README_START,
             "## 0. Full-Dataset Quantitative Evaluation (principal results)\n",
             f"Condition-wise Raider identity-association success on **{cov['videos']} videos / {cov['eligible']} eligible "
             f"raider frames** of `dhruba_dataset_processed` (reviewed GT, all heuristics on the same GT boxes). "
             f"Full report: `condition_classification/HEURISTIC_CONDITION_SUMMARY.md` (Part 1); frame conditions: "
             f"`condition_classification/classification_summary.md`. Pipeline: `run_full_dataset_evaluation.py`.\n"]
    block += _success_table(shc, ctx["table"], ctx["same"])
    block.append("\n\\* " + "; ".join(f"{h} = {shc.implemented_as(h, ctx['same'])}" for h in shc.starred(ctx["same"])) + ".\n")
    block.append("Sections 1–6 below describe the **C01–C19 qualitative diagnostic study** (selected windows, case-level "
                 "S/P/F statuses and switch counts). Those case-level metrics are a different population and metric from "
                 "the table above and must not be compared directly. Note: the E7/E8 rows in sections 6 were produced "
                 "before the 2026-10-13 E7/E8 fix (see `condition_classification/HEURISTIC_CONDITION_SUMMARY.md`).\n")
    block.append(README_END)
    text = path.read_text(encoding="utf-8")
    new = "\n".join(block)
    if README_START in text:
        text = text[:text.index(README_START)] + new + text[text.index(README_END) + len(README_END):]
    else:
        anchor = text.index("---\n", text.index("**Scope**")) + 4
        text = text[:anchor] + "\n" + new + "\n\n---\n" + text[anchor:]
    path.write_text(text, encoding="utf-8")
