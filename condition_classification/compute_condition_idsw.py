"""
compute_condition_idsw.py
=========================
Calculates condition-wise Identity Switches (IDSW) for all tracking heuristics
(E0, E2, E3, E5, E7, E8) across the full reviewed dataset using the saved
per-frame associations and ground-truth condition classifications.

Data source:
    diagnostic_analysis/condition_classification/frame_heuristic_association.csv

Definitions:
    - An ID switch is counted when the heuristic's assigned track ID on the GT raider
      changes between consecutive observed raider frames:
          current_tid != last_tid (where last_tid is the previous valid raider track ID).
    - Condition attribution: each switch is attributed to the condition label at the
      transition frame (the frame where the switch is first observed).
    - Unclassifiable frames (e.g. depth UNKNOWN/NOT_APPLICABLE, speed UNKNOWN, overlap NOT_APPLICABLE)
      are not assigned to condition columns, but all valid switches are included in 'All Eligible'.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ASSOC_CSV = SCRIPT_DIR / "frame_heuristic_association.csv"
if not ASSOC_CSV.exists():
    ASSOC_CSV = SCRIPT_DIR / "condition_classification" / "frame_heuristic_association.csv"

HEURISTICS = [
    ("E0", "E0 (ByteTrack)"),
    ("E2", "E2 (Torso ROI)"),
    ("E3", "E3 (Pose Align*)"),
    ("E5", "E5 (Hip Colour)"),
    ("E7", "E7 (SAM Mask)"),
    ("E8", "E8 (MiDaS Depth)"),
]


def load_associations() -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    data: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with open(ASSOC_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["heuristic"]][row["case_id"]].append(row)
    return data


def compute_idsw_tables(data: Dict[str, Dict[str, List[Dict[str, str]]]]) -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}

    for h_code, h_name in HEURISTICS:
        switches: List[Tuple[Dict[str, str], Dict[str, str]]] = []
        for case_id, rows in data[h_code].items():
            rows.sort(key=lambda r: int(r["frame_number"]))
            last_row = None
            for r in rows:
                tid = r["raider_heuristic_track_id"]
                if not tid:  # missing prediction: track dropped out, no ID assigned
                    continue
                if last_row is None:
                    # Initial track birth: establishes initial ID, not a switch
                    last_row = r
                elif tid != last_row["raider_heuristic_track_id"]:
                    # Identity switch transition
                    switches.append((last_row, r))
                    last_row = r

        counts = {
            "Very High Overlap": 0,
            "High Overlap": 0,
            "Low Overlap": 0,
            "High Speed": 0,
            "Low Speed": 0,
            "Raider in Front": 0,
            "Raider Behind": 0,
            "All Eligible": len(switches),
            "Overlap N/A": 0,
            "Speed Unknown": 0,
            "Depth Unknown/NA": 0,
        }
        
        switch_vids = defaultdict(set)
        for prev_r, curr_r in switches:
            cid = curr_r["case_id"]
            switch_vids["All Eligible"].add(cid)
            
            ov = curr_r["overlap_label"]
            if ov == "VERY_HIGH_OVERLAP":
                counts["Very High Overlap"] += 1
                switch_vids["Very High Overlap"].add(cid)
            elif ov == "HIGH_OVERLAP":
                counts["High Overlap"] += 1
                switch_vids["High Overlap"].add(cid)
            elif ov == "LOW_OVERLAP":
                counts["Low Overlap"] += 1
                switch_vids["Low Overlap"].add(cid)
            else:
                counts["Overlap N/A"] += 1

            sp = curr_r["speed_label"]
            if sp == "HIGH_SPEED":
                counts["High Speed"] += 1
                switch_vids["High Speed"].add(cid)
            elif sp == "LOW_SPEED":
                counts["Low Speed"] += 1
                switch_vids["Low Speed"].add(cid)
            else:
                counts["Speed Unknown"] += 1

            dp = curr_r["depth_label"]
            if dp == "FRONT":
                counts["Raider in Front"] += 1
                switch_vids["Raider in Front"].add(cid)
            elif dp == "BEHIND":
                counts["Raider Behind"] += 1
                switch_vids["Raider Behind"].add(cid)
            else:
                counts["Depth Unknown/NA"] += 1

        results[h_code] = {
            "counts": counts,
            "switch_vids": {k: len(v) for k, v in switch_vids.items()},
        }

    return results


def get_condition_totals() -> Tuple[Dict[str, int], Dict[str, int]]:
    cond_vids = defaultdict(set)
    cond_frames = defaultdict(int)
    with open(ASSOC_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["heuristic"] == "E0":
                cid = r["case_id"]
                cond_vids["All Eligible"].add(cid)
                cond_frames["All Eligible"] += 1
                ov = r["overlap_label"]
                if ov == "VERY_HIGH_OVERLAP":
                    cond_vids["Very High Overlap"].add(cid); cond_frames["Very High Overlap"] += 1
                elif ov == "HIGH_OVERLAP":
                    cond_vids["High Overlap"].add(cid); cond_frames["High Overlap"] += 1
                elif ov == "LOW_OVERLAP":
                    cond_vids["Low Overlap"].add(cid); cond_frames["Low Overlap"] += 1
                
                sp = r["speed_label"]
                if sp == "HIGH_SPEED":
                    cond_vids["High Speed"].add(cid); cond_frames["High Speed"] += 1
                elif sp == "LOW_SPEED":
                    cond_vids["Low Speed"].add(cid); cond_frames["Low Speed"] += 1
                    
                dp = r["depth_label"]
                if dp == "FRONT":
                    cond_vids["Raider in Front"].add(cid); cond_frames["Raider in Front"] += 1
                elif dp == "BEHIND":
                    cond_vids["Raider Behind"].add(cid); cond_frames["Raider Behind"] += 1
                    
    return {k: len(v) for k, v in cond_vids.items()}, dict(cond_frames)


def main():
    print(f"Loading associations from: {ASSOC_CSV}")
    data = load_associations()
    results = compute_idsw_tables(data)
    tot_vids, tot_frames = get_condition_totals()

    conditions = [
        "Very High Overlap", "High Overlap", "Low Overlap",
        "High Speed", "Low Speed",
        "Raider in Front", "Raider Behind",
        "All Eligible",
    ]

    print("\n" + "=" * 95)
    print("FORMAT: switches (affected_videos / total_videos in condition)")
    print("=" * 95)
    head = ["Heuristic"] + conditions
    print("| " + " | ".join(head) + " |")
    print("|---|" + "---:|" * len(conditions))
    for h_code, h_name in HEURISTICS:
        cells = []
        for c in conditions:
            sw = results[h_code]["counts"][c]
            sv = results[h_code]["switch_vids"].get(c, 0)
            tv = tot_vids[c]
            cells.append(f"{sw} ({sv}/{tv} vids)")
        print(f"| {h_name} | " + " | ".join(cells) + " |")

    print("\n" + "=" * 95)
    print("FORMAT: switches / total_videos (e.g. 10/87 vids)")
    print("=" * 95)
    print("| " + " | ".join(head) + " |")
    print("|---|" + "---:|" * len(conditions))
    for h_code, h_name in HEURISTICS:
        cells = []
        for c in conditions:
            sw = results[h_code]["counts"][c]
            tv = tot_vids[c]
            cells.append(f"{sw}/{tv} vids")
        print(f"| {h_name} | " + " | ".join(cells) + " |")

    print("\n" + "=" * 95)
    print("FORMAT: switch_rate% (switches / total_frames) (matching Table 1, e.g. 4.0% (10/252))")
    print("=" * 95)
    print("| " + " | ".join(head) + " |")
    print("|---|" + "---:|" * len(conditions))
    for h_code, h_name in HEURISTICS:
        cells = []
        for c in conditions:
            sw = results[h_code]["counts"][c]
            tf = tot_frames[c]
            pct = (sw / tf) * 100.0
            cells.append(f"{pct:.1f}% ({sw}/{tf})")
        print(f"| {h_name} | " + " | ".join(cells) + " |")


    print("\nPartition Verification:")
    for h_code, h_name in HEURISTICS:
        c = results[h_code]["counts"]
        tot = c["All Eligible"]
        ov_sum = c["Very High Overlap"] + c["High Overlap"] + c["Low Overlap"] + c["Overlap N/A"]
        sp_sum = c["High Speed"] + c["Low Speed"] + c["Speed Unknown"]
        dp_sum = c["Raider in Front"] + c["Raider Behind"] + c["Depth Unknown/NA"]
        assert ov_sum == tot, f"{h_code} overlap partition mismatch"
        assert sp_sum == tot, f"{h_code} speed partition mismatch"
        assert dp_sum == tot, f"{h_code} depth partition mismatch"
        print(f"  {h_code}: {tot} total switches | Overlap sum: {ov_sum} | Speed sum: {sp_sum} | Depth sum: {dp_sum} [OK]")


if __name__ == "__main__":
    main()
