"""
compare_heuristic_tracks.py
===========================
Exact and semantic comparison of tracks_E0/E3/E7/E8.json for the condition-classification
cases. Used by the E7/E8 implementation audit (HEURISTIC_IMPLEMENTATION_AUDIT.md).

For each case and heuristic pair it reports:
  - byte-identical files / identical parsed JSON
  - frame count and per-frame detection count equality
  - max |bbox difference| (GT boxes are shared inputs, so this should be 0)
  - frames where any track_id differs (raw label comparison)
  - whether assignments are identical up to a relabelling of track IDs
    (a consistent one-to-one ID mapping exists across the whole window)
  - frames where the raider's track_id differs

Usage:
    python diagnostic_analysis/compare_heuristic_tracks.py [--out path.json] [--md path.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
CASES = ["C01", "C03", "C04", "C05", "C06", "C07", "C10", "C14", "C19"]
HEURISTICS = ["E0", "E3", "E7", "E8"]


def compare(a_path: Path, b_path: Path) -> Dict[str, Any]:
    ra, rb = a_path.read_bytes(), b_path.read_bytes()
    a, b = json.loads(ra), json.loads(rb)
    res: Dict[str, Any] = {
        "byte_identical": hashlib.sha256(ra).digest() == hashlib.sha256(rb).digest(),
        "parsed_identical": a == b,
        "n_frames": [len(a), len(b)],
        "frame_names_equal": [f["frame"] for f in a] == [f["frame"] for f in b],
    }
    det_count_diff, bbox_max, id_diff_frames, raider_diff_frames = [], 0.0, [], []
    fwd: Dict[Any, Any] = {}
    bwd: Dict[Any, Any] = {}
    relabel_consistent = True
    for fa, fb in zip(a, b):
        ta, tb = fa["tracks"], fb["tracks"]
        if len(ta) != len(tb):
            det_count_diff.append(fa["frame"])
            continue
        any_id = False
        for x, y in zip(ta, tb):
            bbox_max = max(bbox_max, max(abs(p - q) for p, q in zip(x["bbox_xyxy"], y["bbox_xyxy"])))
            if x["track_id"] != y["track_id"]:
                any_id = True
            if fwd.setdefault(x["track_id"], y["track_id"]) != y["track_id"] or \
               bwd.setdefault(y["track_id"], x["track_id"]) != x["track_id"]:
                relabel_consistent = False
            if x["class_name"] == "raider" and x["track_id"] != y["track_id"]:
                raider_diff_frames.append(fa["frame"])
        if any_id:
            id_diff_frames.append(fa["frame"])
    res.update(
        frames_with_detection_count_diff=det_count_diff,
        max_abs_bbox_diff=round(bbox_max, 6),
        frames_with_any_track_id_diff=len(id_diff_frames),
        identical_up_to_id_relabelling=relabel_consistent,
        frames_with_raider_track_id_diff=sorted(set(raider_diff_frames)),
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    out: Dict[str, Any] = {}
    for cid in CASES:
        for h1, h2 in combinations(HEURISTICS, 2):
            out[f"{cid}:{h1}-vs-{h2}"] = compare(SCRIPT_DIR / cid / f"tracks_{h1}.json",
                                                 SCRIPT_DIR / cid / f"tracks_{h2}.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if args.md:
        L: List[str] = ["| Case | Pair | Byte-identical | Parsed identical | Frames | Det. count diffs | "
                        "Max bbox diff | Frames with any ID diff | Same up to relabelling | Raider-ID diff frames |",
                        "|---|---|---|---|---|---:|---:|---:|---|---:|"]
        for k, v in out.items():
            cid, pair = k.split(":")
            L.append(f"| {cid} | {pair} | {v['byte_identical']} | {v['parsed_identical']} | "
                     f"{v['n_frames'][0]}/{v['n_frames'][1]} | {len(v['frames_with_detection_count_diff'])} | "
                     f"{v['max_abs_bbox_diff']} | {v['frames_with_any_track_id_diff']} | "
                     f"{v['identical_up_to_id_relabelling']} | {len(v['frames_with_raider_track_id_diff'])} |")
        args.md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
