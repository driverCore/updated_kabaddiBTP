"""
summarize_heuristic_conditions.py
=================================
Condition-wise diagnostic evaluation of the EXISTING heuristic outputs (E0, E2, E3,
E5, E7, E8) on the nine cases labelled by classify_frame_conditions.py.

Nothing is re-run: heuristic association results are read from the existing
diagnostic_analysis/<case>/tracks_<EX>.json files produced by generate_gt_tool_cases.py,
and frame conditions from condition_classification/all_cases_frame_conditions.csv.

Raider identity-association success (per eligible frame, per heuristic)
-----------------------------------------------------------------------
Every tracks_<EX>.json box carries the fixed GT box/identity (`original_id`, `bbox_xyxy`)
plus the heuristic's assigned `track_id`. For each case and heuristic:

  matched ID  = the heuristic track_id that sits on the GT raider box in the most eligible
                frames of the case window. For a single GT trajectory this is the IDF1
                identity match (Ristani et al., 2016): the tracker ID maximising ID true positives.
  frame correct  <=>  track_id on the GT raider box == matched ID
                      AND no other GT box in that frame carries the matched ID

Consequences: after an ID switch, the frames on the minority side of the switch are
incorrect; a lost track re-born under a new ID is incorrect; the raider taking a
defender's ID, or a defender taking the raider's matched ID, is incorrect. Frames without
an eligible GT raider (see the classification CSV) are excluded from every denominator.

Why not anchor on the first frame: in C04, C05, C06, C07 and C10 the window starts at the
first GT frame of the sequence, where generate_gt_tool_cases.py emits the GT ID as a
placeholder for newborn E2/E3/E7 tracks (the documented "birth-frame" artifact). A
first-frame anchor would then mark the whole case as failed although the heuristic held
one ID throughout. The first-frame anchor is still reported as a sensitivity variant, and
frames with the placeholder signature are flagged.

Usage:
    python diagnostic_analysis/summarize_heuristic_conditions.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
COND_DIR = SCRIPT_DIR / "condition_classification"
HEURISTICS = ["E0", "E2", "E3", "E5", "E7", "E8"]
HEURISTIC_LABELS = {
    "E0": "ByteTrack",
    "E2": "Torso ROI",
    "E3": "Pose Align",
    "E5": "Hip Colour",
    "E7": "SAM Mask",
    "E8": "MiDaS Depth",
}
CONDITIONS = [
    ("overlap_label", "HIGH_OVERLAP", "High overlap"),
    ("overlap_label", "LOW_OVERLAP", "Low overlap"),
    ("speed_label", "HIGH_SPEED", "High speed"),
    ("speed_label", "LOW_SPEED", "Low speed"),
    ("depth_label", "FRONT", "Raider in front"),
    ("depth_label", "BEHIND", "Raider behind"),
]
ALL = ("eligible", "1", "All eligible")
BOX_TOL = 0.006          # classification CSV stores raider boxes rounded to 2 dp
VERY_HIGH_IOU = 0.5
SMALL_N = 20             # cells with fewer eligible frames are flagged as small
MIN_CASES = 3            # cells drawn from fewer cases are flagged as case-concentrated


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def pct(k: int, n: int) -> Optional[float]:
    return 100.0 * k / n if n else None


def load_heuristic(case_id: str, h: str) -> Dict[str, Dict[str, Any]]:
    recs = json.loads((SCRIPT_DIR / case_id / f"tracks_{h}.json").read_text(encoding="utf-8"))
    return {r["frame"]: r for r in recs}


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def find_raider_box(rec: Dict[str, Any], row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    target = [float(row[k]) for k in ("raider_x1", "raider_y1", "raider_x2", "raider_y2")]
    cands = [t for t in rec["tracks"] if t.get("class_name") == "raider"
             and int(t["original_id"]) == int(row["raider_gt_id"])
             and all(abs(a - b) <= BOX_TOL for a, b in zip(t["bbox_xyxy"], target))]
    return cands[0] if len(cands) == 1 else None


def harness_switch_count(seq: List[Optional[Any]]) -> int:
    """Same rule as generate_gt_tool_cases.py: count changes of the raider's ID between frames."""
    last, n = None, 0
    for tid in seq:
        if tid is None:
            continue
        if last is None:
            last = tid
        elif tid != last:
            n += 1
            last = tid
    return n


def load_diagnostic_rows() -> List[Dict[str, str]]:
    """Frame conditions of the selected diagnostic cases (per-case CSVs, kept unchanged)."""
    rows: List[Dict[str, str]] = []
    for p in sorted(COND_DIR.glob("C[0-9]*/frame_conditions.csv")):
        for row in csv.DictReader(p.open(encoding="utf-8")):
            if (row["eligible"] == "1" and row["overlap_label"] == "HIGH_OVERLAP"
                    and row["max_overlap_iou"] not in ("", None)
                    and float(row["max_overlap_iou"]) > VERY_HIGH_IOU):
                row["overlap_label"] = "VERY_HIGH_OVERLAP"
            rows.append(row)
    return rows


def judge_frames(per_frame: List[Dict[str, Any]], vid: str, h: str, validation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    per_frame: eligible frames of one video/case for one heuristic, in frame order, each
               {"row": condition row, "tid": raider's heuristic id, "ids": all ids in frame, "ridx": raider index}
    Applies the matched-ID (IDF1-style) rule and the first-frame-anchor sensitivity rule.
    """
    out: List[Dict[str, Any]] = []
    if not per_frame:
        return out
    first_anchor = per_frame[0]["tid"]
    # Missing tracker output is a failure, never an identity anchor.
    observed_ids = [pf["tid"] for pf in per_frame if pf["tid"] is not None]
    majority_anchor = Counter(observed_ids).most_common(1)[0][0] if observed_ids else None
    p0 = per_frame[0]
    placeholder_frame = (p0["row"]["frame_number"] if len(per_frame) > 1
                         and str(p0["tid"]) == str(p0["row"]["raider_gt_id"])
                         and per_frame[1]["tid"] != p0["tid"] else None)
    if placeholder_frame:
        validation.setdefault("birth_placeholder_suspects", []).append(f"{vid}/{h}/frame {placeholder_frame}")
    for pf in per_frame:
        r, tid, ids, ridx = pf["row"], pf["tid"], pf["ids"], pf["ridx"]

        def judge(anchor: Any) -> Tuple[int, str]:
            if tid is None:
                return 0, "MISSING_RAIDER_PREDICTION"
            if anchor is None:
                return 0, "NO_MATCHED_RAIDER_TRACK"
            holders = [i for i, x in enumerate(ids) if x == anchor]
            if tid != anchor:
                return 0, "ANCHOR_ID_ON_OTHER_BOX" if any(i != ridx for i in holders) else "RAIDER_ID_DIFFERS_FROM_ANCHOR"
            if len(holders) > 1:
                return 0, "ANCHOR_ID_DUPLICATED_IN_FRAME"
            return 1, "CORRECT"

        c1, why1 = judge(first_anchor)
        c2, why2 = judge(majority_anchor)
        out.append({
            "case_id": vid, "frame_number": r["frame_number"], "frame_name": r["frame_name"],
            "heuristic": h, "raider_gt_id": r["raider_gt_id"], "raider_heuristic_track_id": tid,
            "matched_id": majority_anchor, "correct": c2, "reason": why2,
            "birth_placeholder_suspect": int(str(r["frame_number"]) == str(placeholder_frame)),
            "anchor_id_first_frame": first_anchor, "correct_first_frame_anchor": c1, "reason_first_frame_anchor": why1,
            "overlap_label": r["overlap_label"], "speed_label": r["speed_label"], "depth_label": r["depth_label"],
        })
    return out


def build_diagnostic() -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any], Dict[str, bool]]:
    """Association verdicts for the selected diagnostic cases from diagnostic_analysis/<case>/tracks_<EX>.json."""
    rows = load_diagnostic_rows()
    cases = sorted({r["case_id"] for r in rows})
    summary_json = json.loads((SCRIPT_DIR / "experiment_results_summary.json").read_text(encoding="utf-8"))
    harness = {c["case_id"]: c for c in summary_json["cases"]}
    validation: Dict[str, Any] = {"cases": cases, "missing_heuristic_files": [], "box_match_failures": [],
                                  "gt_box_set_mismatch_frames": [], "switch_count_check": {}}
    identical = {"E3==E7": True, "E0==E8": True}
    assoc: List[Dict[str, Any]] = []
    for cid in cases:
        crow = sorted((r for r in rows if r["case_id"] == cid), key=lambda r: int(r["frame_number"]))
        gt = load_heuristic(cid, "GT")
        for a, b in (("E3", "E7"), ("E0", "E8")):
            if file_hash(SCRIPT_DIR / cid / f"tracks_{a}.json") != file_hash(SCRIPT_DIR / cid / f"tracks_{b}.json"):
                identical[f"{a}=={b}"] = False
        for h in HEURISTICS:
            p = SCRIPT_DIR / cid / f"tracks_{h}.json"
            if not p.is_file():
                validation["missing_heuristic_files"].append(str(p))
                continue
            hrec = load_heuristic(cid, h)
            per_frame: List[Dict[str, Any]] = []
            all_raider_ids: List[Optional[Any]] = []
            for r in crow:
                rec = hrec.get(r["frame_name"])
                if rec is not None and r["eligible"] == "1":
                    tb = find_raider_box(rec, r)
                    all_raider_ids.append(tb["track_id"] if tb else None)
                else:
                    rs = [t["track_id"] for t in (rec or {}).get("tracks", []) if t.get("class_name") == "raider"]
                    all_raider_ids.append(rs[0] if rs else None)
                if r["eligible"] != "1":
                    continue
                if rec is None:
                    validation["box_match_failures"].append(f"{cid}/{h}/{r['frame_name']}: frame missing")
                    continue
                g = gt[r["frame_name"]]
                key = lambda rr: sorted((int(t["original_id"]), tuple(round(v, 3) for v in t["bbox_xyxy"])) for t in rr["tracks"])
                if key(g) != key(rec):
                    validation["gt_box_set_mismatch_frames"].append(f"{cid}/{h}/{r['frame_name']}")
                tb = find_raider_box(rec, r)
                if tb is None:
                    validation["box_match_failures"].append(f"{cid}/{h}/{r['frame_name']}: raider box not found")
                    continue
                ids = [t["track_id"] for t in rec["tracks"]]
                per_frame.append({"row": r, "tid": tb["track_id"], "ids": ids,
                                  "ridx": next(i for i, t in enumerate(rec["tracks"]) if t is tb)})
            hs = harness_switch_count(all_raider_ids)
            validation["switch_count_check"][f"{cid}/{h}"] = {
                "recomputed": hs, "experiment_results_summary": harness[cid]["switches"][h],
                "match": hs == harness[cid]["switches"][h]}
            assoc.extend(judge_frames(per_frame, cid, h, validation))
    validation["identical_outputs"] = identical
    validation["all_switch_counts_match"] = all(v["match"] for v in validation["switch_count_check"].values())
    return rows, assoc, validation, identical


def cell(assoc: List[Dict[str, Any]], h: str, field: str, value: str, col: str = "correct") -> Dict[str, Any]:
    sel = [a for a in assoc if a["heuristic"] == h and (field == "eligible" or a[field] == value)]
    k, n = sum(a[col] for a in sel), len(sel)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "pct": pct(k, n), "lo": lo, "hi": hi,
            "cases": len({a["case_id"] for a in sel}),
            "case_breakdown": Counter(a["case_id"] for a in sel)}


def compute_tables(assoc: List[Dict[str, Any]]):
    """Returns (table, cell_rows, by_case_rows) for the given association verdicts."""
    cell_rows = []
    table: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for h in HEURISTICS:
        for field, value, label in CONDITIONS + [ALL]:
            c1 = cell(assoc, h, field, value)
            c2 = cell(assoc, h, field, value, "correct_first_frame_anchor")
            table[h][label] = c1
            cell_rows.append({
                "heuristic": h, "factor": field, "condition": value if field != "eligible" else "ALL_ELIGIBLE",
                "column_label": label, "n_correct": c1["k"], "n_eligible_frames": c1["n"],
                "success_rate_pct": "" if c1["pct"] is None else round(c1["pct"], 2),
                "wilson95_lo_pct": "" if c1["lo"] is None else round(c1["lo"], 2),
                "wilson95_hi_pct": "" if c1["hi"] is None else round(c1["hi"], 2),
                "n_cases_contributing": c1["cases"],
                "frames_per_case": ";".join(f"{k}:{v}" for k, v in sorted(c1["case_breakdown"].items())),
                "small_cell_flag": int(c1["n"] < SMALL_N or c1["cases"] < MIN_CASES),
                "n_correct_first_frame_anchor": c2["k"],
                "success_rate_pct_first_frame_anchor": "" if c2["pct"] is None else round(c2["pct"], 2),
            })
    by_case = []
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in assoc:
        groups[a["case_id"]].append(a)
    for cid in sorted(groups):
        sub = groups[cid]
        for h in HEURISTICS:
            for field, value, label in CONDITIONS + [ALL]:
                c = cell(sub, h, field, value)
                if c["n"]:
                    by_case.append({"case_id": cid, "heuristic": h, "column_label": label,
                                    "n_correct": c["k"], "n_eligible_frames": c["n"],
                                    "success_rate_pct": round(c["pct"], 2)})
    return table, cell_rows, by_case


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(path: Path, table, identical: Dict[str, bool]) -> None:
    labels = [c[2] for c in CONDITIONS]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["heuristic", "implemented_as"] + [f"{l} (%)" for l in labels] + [f"{l} (n/N)" for l in labels]
                   + ["All eligible (%)", "All eligible (n/N)"])
        for h in HEURISTICS:
            t = table[h]
            w.writerow([h, implemented_as(h, identical)]
                       + [fmt_pct(t[l]) for l in labels] + [f"{t[l]['k']}/{t[l]['n']}" for l in labels]
                       + [fmt_pct(t["All eligible"]), f"{t['All eligible']['k']}/{t['All eligible']['n']}"])


def table_md(table, identical) -> List[str]:
    labels = [c[2] for c in CONDITIONS]
    L = ["| Heuristic | " + " | ".join(labels) + " | All eligible |", "|---|" + "---:|" * (len(labels) + 1)]
    for h in HEURISTICS:
        t = table[h]
        cells = [f"{fmt_pct(t[l])} ({t[l]['k']}/{t[l]['n']})" for l in labels + ["All eligible"]]
        star = "*" if h in starred(identical) else ""
        L.append(f"| {h} ({HEURISTIC_LABELS[h]}{star}) | " + " | ".join(cells) + " |")
    return L


def main() -> None:
    print("The aggregate files in condition_classification/ now hold the FULL-DATASET evaluation and are written by\n"
          "    python diagnostic_analysis/run_full_dataset_evaluation.py evaluate\n"
          "which also renders the C01-C19 diagnostic section from this module (build_diagnostic).")
    rows, assoc, validation, identical = build_diagnostic()
    table, _, _ = compute_tables(assoc)
    print("\n".join(table_md(table, identical)))
    print("diagnostic switch counts match experiment_results_summary.json:", validation["all_switch_counts_match"])


def implemented_as(h: str, identical: Dict[str, bool]) -> str:
    return {
        "E0": "ByteTrack (Kalman + 2-stage IoU)",
        "E2": "greedy torso-crop IoU",
        "E3": "greedy full-box IoU (associate_by_iou); no pose keypoints" + (" — identical output to E7" if identical["E3==E7"] else ""),
        "E5": "HSV lower-body colour team tracker",
        "E7": ("greedy full-box IoU (associate_by_iou); no SAM masks — identical output to E3" if identical["E3==E7"]
               else "greedy MobileSAM mask-IoU association (associate_by_mask_iou)"),
        "E8": ("copy of E0 ByteTrack assignments; no MiDaS — identical output to E0" if identical["E0==E8"]
               else "Hungarian on (1-IoU) + 0.01*|MiDaS depth diff| (associate_by_iou_depth)"),
    }[h]


def fmt_pct(c: Dict[str, Any]) -> str:
    return "n/a" if c["pct"] is None else f"{c['pct']:.1f}%"


def render_md(rows, assoc, table, validation, identical) -> str:
    labels = [c[2] for c in CONDITIONS]
    elig = [r for r in rows if r["eligible"] == "1"]
    L: List[str] = []
    L.append("# Condition-Wise Heuristic Performance — Diagnostic Evaluation on Selected Cases\n")
    L.append("> **Scope:** condition-wise diagnostic evaluation on 9 selected cases (C01, C03, C04, C05, C06, C07, "
             "C10, C14, C19; 227 window frames). This is **not** a full-dataset benchmark. Heuristic outputs are the "
             "`tracks_EX.json` files as currently stored (this script runs no tracker).\n")
    L.append("> [!WARNING]\n> " + cue_warning(identical) + "\n")

    L.append("## Raider identity-association success rate by frame condition\n")
    L.append("| Heuristic | " + " | ".join(labels) + " | All eligible |")
    L.append("|---|" + "---:|" * (len(labels) + 1))
    for h in HEURISTICS:
        t = table[h]
        cells = [f"{fmt_pct(t[l])} ({t[l]['k']}/{t[l]['n']})" for l in labels + ["All eligible"]]
        star = "*" if h in starred(identical) else ""
        L.append(f"| {h} ({HEURISTIC_LABELS[h]}{star}) | " + " | ".join(cells) + " |")
    L.append("\n\\* " + "; ".join(f"{h} = {implemented_as(h, identical)}" for h in starred(identical)) + ".\n")
    L.append("Success rate = correctly associated eligible raider frames / eligible raider frames in that condition × 100. "
             "Frames are pooled across cases (micro-average).\n")

    L.append("### Success definition\n")
    L.append("- Reference: fixed GT raider box and GT raider identity in each frame (the heuristics were fed these GT boxes).")
    L.append("- **Matched ID** (per case × heuristic) = the heuristic `track_id` found on the GT raider box in the most "
             "eligible frames of the window — the IDF1 identity match for a single GT trajectory.")
    L.append("- A frame is **correct** iff the GT raider box carries the matched ID and no other box in that frame does.")
    L.append("- **ID switch**: frames on the minority side of the switch (the ID that is not the matched ID) are incorrect.")
    L.append("- **Lost track / re-birth**: the raider re-appears under a new ID → incorrect unless that new ID becomes the "
             "majority ID, in which case the earlier frames are the incorrect ones.")
    L.append("- **Incorrect raider association**: the raider takes another player's ID, or another box takes the matched ID → incorrect.")
    L.append("- Frames without an eligible GT raider are excluded from all denominators (they are not counted as failures or successes).")
    ph = validation.get("birth_placeholder_suspects", [])
    L.append(f"- **Why not the first-frame ID:** in C04, C05, C06, C07 and C10 the window starts at the first GT frame, "
             f"where the harness writes the GT ID as a placeholder whenever a heuristic returns no ID for a box "
             f"(e.g. newborn E2/E3/E7 tracks; documented birth-frame artifact). Frames with that signature (first frame = GT ID, next frame different): "
             f"{len(ph)} — {', '.join(ph) or 'none'}. A first-frame anchor would mark those whole cases failed; it is "
             f"reported below as a sensitivity check only.")
    L.append("- Per-frame verdicts: `frame_heuristic_association.csv`; counts: `summary_cell_counts.csv` "
             "(with Wilson 95% intervals and per-case frame counts); per-case split: `summary_cell_counts_by_case.csv`.\n")

    # Supporting information
    dc = Counter(r["depth_label"] for r in elig)
    L.append("## Supporting information\n")
    L.append("| Condition | Eligible frames | Cases contributing (frames per case) |\n|---|---:|---|")
    for l in labels + ["All eligible"]:
        c = table["E0"][l]
        L.append(f"| {l} | {c['n']} | {c['cases']} ({', '.join(f'{k}:{v}' for k, v in sorted(c['case_breakdown'].items()))}) |")
    L.append("")
    excl = Counter(r["ineligible_reason"] for r in rows if r["eligible"] != "1")
    L.append(f"- **Frames processed:** {len(rows)}; **eligible:** {len(elig)}; **excluded:** {len(rows) - len(elig)} "
             f"({', '.join(f'{k}: {v}' for k, v in excl.items()) or 'none'}). The excluded frames are C05 frames 81–87, "
             "where the source GT has no raider box (a 7-frame raider gap).")
    L.append(f"- **Depth labels:** FRONT {dc['FRONT']}, BEHIND {dc['BEHIND']}, UNKNOWN {dc['UNKNOWN']}, "
             f"NOT_APPLICABLE {dc['NOT_APPLICABLE']}, BLOCKED {dc['BLOCKED']}. UNKNOWN/NOT_APPLICABLE frames are excluded "
             "from the two depth columns only.")
    L.append("- **Depth computed:** yes — existing MiDaS_small implementation (`tracking/E8_midas_depth/depth_features.py`) "
             "on the original JPGs; larger MiDaS value = closer (inverse depth), confirmed per frame "
             "(see `classification_summary.md`).")
    L.append("- **Thresholds:** speed HIGH if raider centre speed > 1050 px/s (= 35 px/frame at 30 fps); depth FRONT/BEHIND if |normalised depth difference| > 0.05. Rationale: `classification_summary.md`, `threshold_configuration.json`.")
    sc = validation["switch_count_check"]
    L.append(f"- **Consistency check:** raider ID-switch counts recomputed from the `tracks_EX.json` files with the "
             f"harness rule match `experiment_results_summary.json` for {sum(v['match'] for v in sc.values())}/{len(sc)} "
             f"case-heuristic pairs; raider box located in every heuristic file for every eligible frame: "
             f"{not validation['box_match_failures']}; GT box sets identical across heuristic files: "
             f"{not validation['gt_box_set_mismatch_frames']}.\n")

    # Sensitivity: first-frame anchor
    L.append("## Sensitivity: first-frame anchor\n")
    L.append("Same frames, anchor = the raider's heuristic ID on the first eligible frame. Every frame after a switch fails "
             "until the ID returns, so the birth-frame placeholder frames listed above fail their entire case.\n")
    L.append("| Heuristic | " + " | ".join(labels) + " | All eligible |")
    L.append("|---|" + "---:|" * (len(labels) + 1))
    for h in HEURISTICS:
        cells = []
        for field, value, l in CONDITIONS + [ALL]:
            sel = [a for a in assoc if a["heuristic"] == h and (field == "eligible" or a[field] == value)]
            k = sum(a["correct_first_frame_anchor"] for a in sel)
            cells.append(f"{fmt_pct({'pct': pct(k, len(sel))})} ({k}/{len(sel)})")
        L.append(f"| {h} | " + " | ".join(cells) + " |")
    L.append("")

    L.append(interpretation(table, labels, identical))
    return "\n".join(L) + "\n"


def starred(identical: Dict[str, bool]) -> List[str]:
    """Heuristics whose output does not use the cue their name refers to."""
    return ["E3"] + (["E7"] if identical["E3==E7"] else []) + (["E8"] if identical["E0==E8"] else [])


def cue_warning(identical: Dict[str, bool]) -> str:
    s = ("**E3 does not implement pose alignment in the evaluated outputs**: `generate_gt_tool_cases.py` calls "
         "greedy full-box IoU (`associate_by_iou`) for E3; no pose keypoints are computed. ")
    if identical["E3==E7"] or identical["E0==E8"]:
        s += ("**E7/E8 also do not use their cues**: E7 calls the same box-IoU function and E8 copies E0 "
              f"(tracks_E3 == tracks_E7: {identical['E3==E7']}; tracks_E0 == tracks_E8: {identical['E0==E8']}). ")
    else:
        s += ("E7 (MobileSAM mask IoU) and E8 (MiDaS depth in a Hungarian cost) were re-implemented in the harness on "
              "2026-10-13 and regenerated for these 9 cases; see `diagnostic_analysis/HEURISTIC_IMPLEMENTATION_AUDIT.md`. ")
    return s + "Rows marked * do not measure the cue in their name."


def interpretation(table, labels, identical) -> str:
    """Auto-generated, number-driven observations (no external claims)."""
    distinct = ["E0", "E2", "E3", "E5"] + ([] if identical["E3==E7"] else ["E7"]) + ([] if identical["E0==E8"] else ["E8"])
    L = ["## Research interpretation (generated from the counts above)\n"]

    def best(label):
        vals = [(table[h][label]["pct"], h) for h in distinct if table[h][label]["pct"] is not None]
        top = max(v for v, _ in vals)
        return top, [h for v, h in vals if abs(v - top) < 1e-9]

    def overlaps_ci(a, b, label):
        ca, cb = table[a][label], table[b][label]
        return not (ca["lo"] > cb["hi"] or cb["lo"] > ca["hi"])

    for label in ("High overlap", "High speed", "Raider behind", "Raider in front"):
        top, who = best(label)
        n = table[who[0]][label]["n"]
        twins = {**({"E0": "E8"} if identical["E0==E8"] else {}), **({"E3": "E7"} if identical["E3==E7"] else {})}
        who_s = ", ".join(w + (f" (= {twins[w]})" if w in twins else "") for w in who)
        others = [h for h in distinct if h not in who and table[h][label]["pct"] is not None]
        indist = [h for h in others if overlaps_ci(who[0], h, label)]
        runner = max(others, key=lambda h: table[h][label]["pct"]) if others else None
        gap = top - table[runner][label]["pct"] if runner else float("nan")
        L.append(f"- **{label}** (N = {n} frames from {table[who[0]][label]['cases']} cases): highest observed rate "
                 f"{who_s} at {top:.1f}%; next best {runner} at {table[runner][label]['pct']:.1f}% "
                 f"(gap {gap:.1f} pp). Wilson 95% intervals overlap with: {', '.join(indist) or 'none'}.")
    # condition difficulty: for each condition, max across heuristics
    hard = []
    for l in labels:
        mx = max(table[h][l]["pct"] for h in distinct if table[h][l]["pct"] is not None)
        hard.append((mx, l))
    hard.sort()
    L.append(f"- **Hardest condition for all heuristics** (lowest best-case rate): {hard[0][1]} "
             f"(best {hard[0][0]:.1f}%), then {hard[1][1]} (best {hard[1][0]:.1f}%).")
    for h in distinct:
        d_ov = table[h]["Low overlap"]["pct"] - table[h]["High overlap"]["pct"]
        d_sp = table[h]["Low speed"]["pct"] - table[h]["High speed"]["pct"]
        d_dp = table[h]["Raider in front"]["pct"] - table[h]["Raider behind"]["pct"]
        L.append(f"- {h}: low−high overlap {d_ov:+.1f} pp; low−high speed {d_sp:+.1f} pp; front−behind {d_dp:+.1f} pp.")
    L.append("- **Cue-specific advantages:** E3 is greedy full-box IoU (no pose), so **no conclusion about pose cues** "
             "can be drawn. " + ("E7 = E3 and E8 = E0, so no conclusion about SAM-mask or depth cues either."
             if identical["E3==E7"] or identical["E0==E8"] else
             "E7 vs E3 isolates the mask cue (same greedy procedure and 0.15 threshold; only box IoU -> mask IoU "
             "changes). E8 vs E0 is NOT a clean depth ablation: E8 uses frame-to-frame Hungarian matching without "
             "ByteTrack's Kalman motion model or 30-frame track buffer, so E8-E0 differences mix the depth term with "
             "the loss of motion prediction."))
    for l in labels:
        cb = table["E0"][l]["case_breakdown"]
        top_case, top_n = cb.most_common(1)[0]
        two = sum(v for _, v in cb.most_common(2))
        L.append(f"- {l}: largest single-case share {top_case} {top_n}/{table['E0'][l]['n']} "
                 f"({100 * top_n / table['E0'][l]['n']:.0f}%); top two cases {100 * two / table['E0'][l]['n']:.0f}%.")
    L.append("- **Case confounding:** each case contributes one identity outcome (e.g. one switch) that is spread across "
             "its frames, and conditions cluster by case (shares above). A condition-level difference can therefore reflect which cases fall in that "
             "condition rather than the condition itself. See `summary_cell_counts_by_case.csv`.")
    small = [(h, l) for h in distinct for l in labels
             if table[h][l]["n"] < SMALL_N or table[h][l]["cases"] < MIN_CASES]
    if small:
        L.append(f"- **Small / case-concentrated cells** (N < {SMALL_N} or < {MIN_CASES} cases): "
                 + ", ".join(sorted({l for _, l in small})) + ".")
    L.append("- Wilson intervals treat frames as independent; consecutive frames of one case are strongly correlated "
             "(a single switch changes the verdict of many consecutive frames), so the true uncertainty is larger than shown.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
