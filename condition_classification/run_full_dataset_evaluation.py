"""
run_full_dataset_evaluation.py
==============================
Full-dataset version of the validated condition-wise evaluation
(classify_frame_conditions.py + summarize_heuristic_conditions.py), applied to every
sequence in dhruba_dataset_processed that has GT tracks.

The methodology is reused, not redesigned:
  - GT detections/identities: dhruba_dataset_processed/<split>/<action>/bytetrack/bytetrack_<seq>_frames/tracks.json
  - Frame conditions: classify_frame_conditions.process_case (same thresholds, same rules),
    with the evaluation window = the whole GT sequence instead of a diagnostic window
  - Trackers: the same per-frame association code as generate_gt_tool_cases.py
    (E0 ByteTrack, E2 torso IoU, E3 associate_by_iou, E5 hip colour, E7 SAM mask IoU,
    E8 MiDaS depth Hungarian), fed the GT boxes, run from the first GT frame
  - Success: summarize_heuristic_conditions (IDF1-style matched ID per video)

Stages (each resumable per video; results cached under condition_classification/cache/):
  sam       SAM (MobileSAM) masks -> consecutive-frame mask-IoU matrices (needs ultralytics; GPU if available)
  classify  frame-condition rows + MiDaS box medians for E8 (needs timm for MiDaS)
  track     run E0/E2/E3/E5/E7/E8 on GT boxes using the cached SAM/MiDaS features
  evaluate  write the condition-wise summary files in condition_classification/ (in place)

Usage:
  python diagnostic_analysis/run_full_dataset_evaluation.py sam        # mmaction_env (CUDA)
  python diagnostic_analysis/run_full_dataset_evaluation.py classify   # Python 3.13 (timm)
  python diagnostic_analysis/run_full_dataset_evaluation.py track
  python diagnostic_analysis/run_full_dataset_evaluation.py evaluate
  Add --limit N / --only <video_id> for partial runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "dhruba_dataset_processed"
CC_DIR = SCRIPT_DIR / "condition_classification"
CACHE = CC_DIR / "cache"
LOG = CACHE / "run_log.jsonl"


# ---------------------------------------------------------------------------
# Dataset enumeration
# ---------------------------------------------------------------------------
def gt_schema(p: Path) -> str:
    """
    'reviewed'  : GT-editor schema (records carry frame_idx; boxes carry status/original_id) — the
                  schema of every file used in the validated diagnostic experiment.
    'raw_tracker': older schema (no frame_idx, no status/original_id, bbox_yolo present): raw
                  YOLO+ByteTrack output never opened in the GT editor -> not GT, excluded.
    """
    d = json.loads(p.read_text(encoding="utf-8"))
    if d and all("frame_idx" in r for r in d) and any("status" in t for r in d for t in r.get("tracks", [])):
        return "reviewed"
    return "raw_tracker"


def enumerate_videos(include_excluded: bool = False) -> List[Dict[str, Any]]:
    out = []
    for tj in sorted(glob.glob(str(DATA_ROOT / "*" / "*" / "bytetrack" / "bytetrack_*_frames" / "tracks.json"))):
        p = Path(tj)
        split, action = p.parents[3].name, p.parents[2].name
        seq = re.fullmatch(r"bytetrack_(.+)_frames", p.parent.name).group(1)
        v = {"video_id": f"{split}/{action}/{seq}", "key": f"{split}__{action}__{seq}",
             "split": split, "action": action, "seq": seq, "gt": p, "schema": gt_schema(p)}
        if include_excluded or v["schema"] == "reviewed":
            out.append(v)
    return out


def frame_no(name: str) -> int:
    return int(re.search(r"(\d+)", name).group(1))


def valid_tracks(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Same box filter as generate_gt_tool_cases.py (bbox_xyxy present with 4 values)."""
    return [t for t in rec.get("tracks", []) if t.get("bbox_xyxy") and len(t["bbox_xyxy"]) == 4]


def log(event: Dict[str, Any]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    event["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def atomic_write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def select_videos(args) -> List[Dict[str, Any]]:
    vids = enumerate_videos()
    if args.only:
        vids = [v for v in vids if v["video_id"] in args.only or v["seq"] in args.only]
    if args.limit:
        vids = vids[: args.limit]
    return vids


# ---------------------------------------------------------------------------
# Stage: SAM mask-IoU cache
# ---------------------------------------------------------------------------
def stage_sam(args) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "tracking" / "E7_sam_mask_iou"))
    from sam_mask_tracker import generate_sam_masks_aligned, masks_to_frame
    import torch
    from ultralytics import SAM
    sam = SAM(str(PROJECT_ROOT / "tracking" / "models" / "mobile_sam.pt"))
    vids = select_videos(args)
    t0, done = time.time(), 0
    for v in vids:
        out = CACHE / f"{v['key']}.sam.json"
        if out.exists():
            continue
        try:
            recs = json.loads(v["gt"].read_text(encoding="utf-8"))
            fdir = DATA_ROOT / v["split"] / v["action"] / "frames" / f"{v['seq']}_frames"
            res = {"video_id": v["video_id"], "pairs": {}, "n_dets": {}}
            prev_masks, prev_name = None, None
            for rec in recs:  # file order, as in generate_gt_tool_cases.py
                tr = valid_tracks(rec)
                if not tr:
                    continue
                img = cv2.imread(str(fdir / f"{rec['frame']}.jpg"))
                if img is None:
                    raise FileNotFoundError(f"{fdir / rec['frame']}.jpg")
                dets = np.array([[float(x) for x in t["bbox_xyxy"]] for t in tr], dtype=np.float32)
                # Same call as generate_sam_masks (bbox prompts); masks kept on the GPU for the IoU
                sr = sam.predict(source=img, bboxes=dets.tolist(), verbose=False)[0]
                if sr.masks is None or len(sr.masks.data) != len(dets):
                    # ultralytics dropped empty-mask prompts: same handling as the E7 tracker
                    aligned, n_empty = generate_sam_masks_aligned(sam, img, dets)
                    md = torch.from_numpy(aligned).to("cuda" if torch.cuda.is_available() else "cpu")
                    res.setdefault("empty_masks", {})[rec["frame"]] = n_empty
                else:
                    md = sr.masks.data
                if tuple(md.shape[1:]) != img.shape[:2]:
                    # rare: fall back to the CPU path of the E7 tracker (masks_to_frame)
                    md = torch.from_numpy(masks_to_frame(md.cpu().numpy(), img.shape[0], img.shape[1])).to(md.device)
                m = md.reshape(len(dets), -1).float()          # (N, H*W) 0/1
                area = m.sum(1)
                res["n_dets"][rec["frame"]] = len(dets)
                if prev_masks is not None:
                    inter = prev_masks @ m.T                    # exact integer pixel counts (< 2^24)
                    union = prev_area[:, None] + area[None, :] - inter
                    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
                    # iou[prev_det][cur_det] (same quantity as mask_iou on boolean masks)
                    res["pairs"][rec["frame"]] = {"prev": prev_name,
                                                  "iou": [[round(float(x), 6) for x in row] for row in iou.cpu()]}
                prev_masks, prev_area, prev_name = m, area, rec["frame"]
            atomic_write(out, res)
            done += 1
            log({"stage": "sam", "video": v["video_id"], "ok": True, "frames": len(res["n_dets"])})
        except Exception as e:
            log({"stage": "sam", "video": v["video_id"], "ok": False, "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
        if done and done % 10 == 0:
            print(f"[sam] {done} videos cached this run, {time.time() - t0:.0f}s", flush=True)
    print("[sam] stage finished", flush=True)


# ---------------------------------------------------------------------------
# Stage: frame conditions (classify_frame_conditions.process_case) + MiDaS medians for E8
# ---------------------------------------------------------------------------
def stage_classify(args) -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    import classify_frame_conditions as cfc

    class FullDepth(cfc.DepthEstimator):
        """Same MiDaS model/maps; also records per-box full-box medians (E8 feature) per frame."""

        def __init__(self):
            super().__init__()
            from depth_features import median_depth_in_bbox
            self.median = median_depth_in_bbox
            self.boxes: Dict[int, List[List[float]]] = {}
            self.medians: Dict[int, List[float]] = {}

        def depth_map(self, img_bgr, frame=None):
            dm = super().depth_map(img_bgr)
            if frame is not None and frame in self.boxes:
                self.medians[frame] = [self.median(dm, b) for b in self.boxes[frame]]
            return dm

    cfg = json.loads(json.dumps(cfc.DEFAULT_CONFIG))
    depth = FullDepth()
    vids = select_videos(args)
    t0, done = time.time(), 0
    for v in vids:
        out = CACHE / f"{v['key']}.classify.json"
        if out.exists():
            continue
        try:
            recs = json.loads(v["gt"].read_text(encoding="utf-8"))
            nums = [frame_no(r["frame"]) for r in recs]
            fdir = DATA_ROOT / v["split"] / v["action"] / "frames" / f"{v['seq']}_frames"
            depth.boxes = {frame_no(r["frame"]): [[float(x) for x in t["bbox_xyxy"]] for t in valid_tracks(r)]
                           for r in recs if valid_tracks(r)}
            depth.medians = {}
            meta = {"seq": v["seq"], "split": v["split"], "action": v["action"],
                    "start_frame": min(nums), "end_frame": max(nums)}
            rows, pre = cfc.process_case(v["video_id"], v["video_id"], meta, cfg, depth, None)
            if pre["issues"] and not rows:
                raise RuntimeError(f"precheck failed: {pre['issues']}")
            # E8 needs medians for every frame with detections, not only classification frames
            by_num = {frame_no(r["frame"]): r for r in recs}
            for n, boxes in depth.boxes.items():
                if n not in depth.medians:
                    img = cv2.imread(str(fdir / f"{by_num[n]['frame']}.jpg"))
                    depth.depth_map(img, frame=n)
            atomic_write(out, {"video_id": v["video_id"], "rows": rows, "pre": pre,
                               "e8_medians": {by_num[n]["frame"]: m for n, m in depth.medians.items()}})
            done += 1
            log({"stage": "classify", "video": v["video_id"], "ok": True, "rows": len(rows),
                 "eligible": sum(r["eligible"] for r in rows)})
        except Exception as e:
            log({"stage": "classify", "video": v["video_id"], "ok": False, "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
        if done and done % 10 == 0:
            print(f"[classify] {done} videos this run, {time.time() - t0:.0f}s", flush=True)
    print("[classify] stage finished", flush=True)


# ---------------------------------------------------------------------------
# Stage: trackers on GT boxes (same per-frame logic as generate_gt_tool_cases.py)
# ---------------------------------------------------------------------------
def greedy_from_matrix(iou_mat: np.ndarray, track_ids: List[Any], n_dets: int, next_id: int, thr: float):
    """Exactly the greedy loop of associate_by_mask_iou, on a precomputed det x track IoU matrix."""
    assigned: Dict[int, Any] = {}
    used: set = set()
    if track_ids:
        iou_mat = iou_mat.copy()
        while True:
            if iou_mat.size == 0 or iou_mat.max() < thr:
                break
            bi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            di, ti = int(bi[0]), int(bi[1])
            if di not in assigned and ti not in used:
                assigned[di] = track_ids[ti]
                used.add(ti)
            iou_mat[di, :] = 0.0
            iou_mat[:, ti] = 0.0
    new_ids = []
    for di in range(n_dets):
        tid = assigned.get(di, next_id)
        if di not in assigned:
            next_id += 1
        new_ids.append(tid)
    return assigned, new_ids, next_id


def track_video(v: Dict[str, Any], sam_cache: Dict[str, Any], e8_medians: Dict[str, List[float]]) -> Dict[str, Any]:
    import generate_gt_tool_cases as gen
    from sam_mask_tracker import IOU_THRESHOLD as E7_THR
    recs = json.loads(v["gt"].read_text(encoding="utf-8"))
    fdir = DATA_ROOT / v["split"] / v["action"] / "frames" / f"{v['seq']}_frames"
    e0_tracker = gen.build_bytetracker()
    e2_prev, e2_next = {}, 1
    e3_prev, e3_next = {}, 1
    e5_tracker = gen.TeamAwareIdentityTracker()
    e7_prev_ids: List[Any] = []   # track id of each previous-frame det (= keys of E7 track_masks, in order)
    e7_prev_frame, e7_next = None, 1
    e8_prev, e8_next = {}, 1
    out: Dict[str, Any] = {}
    for f_idx, rec in enumerate(recs):
        tr = valid_tracks(rec)
        boxes = [[float(x) for x in t["bbox_xyxy"]] for t in tr]
        gt_ids = [int(t.get("track_id", t.get("original_id", 1))) for t in tr]
        n = len(boxes)
        if n == 0:
            continue
        dets = np.array(boxes, dtype=np.float32)
        # E0
        e0 = {}
        mock = gen.MockResults(dets, np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32))
        t_arr = e0_tracker.update(mock, img=None)
        for row in t_arr:
            tb, tid = row[:4], int(row[4])
            best_i, best_iou = -1, 0.4
            for di, gb in enumerate(boxes):
                ov = gen.bbox_iou(tb, gb)
                if ov > best_iou:
                    best_iou, best_i = ov, di
            if best_i >= 0 and best_i not in e0:
                e0[best_i] = tid
        # E2, E3. The helpers only return existing-track matches; create the
        # real sequential ID for a new detection rather than injecting its GT ID.
        e2_start = e2_next
        e2, e2_prev, e2_next = gen.associate_torso_roi(dets, e2_prev, e2_next)
        e2_ids, e2_new = [], e2_start
        for di in range(n):
            if di in e2:
                e2_ids.append(e2[di])
            else:
                e2_ids.append(e2_new)
                e2_new += 1
        e3_start = e3_next
        e3, e3_prev, e3_next = gen.associate_by_iou(dets, e3_prev, e3_next)
        e3_ids, e3_new = [], e3_start
        for di in range(n):
            if di in e3:
                e3_ids.append(e3[di])
            else:
                e3_ids.append(e3_new)
                e3_new += 1
        # E5
        e5 = {}
        img_path = fdir / f"{rec['frame']}.jpg"
        frame_img = cv2.imread(str(img_path)) if img_path.exists() else None
        for di, bbox in enumerate(dets):
            crop = gen.get_lower_body_crop(frame_img, bbox) if frame_img is not None else None
            team, white = gen.classify_team(crop)
            disp, _ = e5_tracker.assign(raw_track_id=-1, observed_team=team, white_ratio=white, bbox=bbox,
                                        frame_number=f_idx, allow_new_track=team is not None)
            if disp is not None:
                e5[di] = disp
        # E7 (cached SAM mask IoU between this frame's and the previous detection frame's masks)
        if e7_prev_frame is None:
            mat = np.zeros((n, 0))
        else:
            pair = sam_cache["pairs"][rec["frame"]]
            if pair["prev"] != e7_prev_frame:
                raise RuntimeError(f"SAM cache order mismatch at {rec['frame']}")
            mat = np.array(pair["iou"], dtype=float).T  # det x prev-det
        e7, e7_ids, e7_next = greedy_from_matrix(mat, e7_prev_ids, n, e7_next, E7_THR)
        e7_prev_ids = e7_ids
        e7_prev_frame = rec["frame"]
        # E8
        d = np.array(e8_medians[rec["frame"]], dtype=float)
        if len(d) != n:
            raise RuntimeError(f"E8 depth cache size mismatch at {rec['frame']}")
        e8_start = e8_next
        e8, e8_prev, e8_next = gen.associate_by_iou_depth(dets, d, e8_prev, e8_next)
        e8_ids, e8_new = [], e8_start
        for di in range(n):
            if di in e8:
                e8_ids.append(e8[di])
            else:
                e8_ids.append(e8_new)
                e8_new += 1
        out[rec["frame"]] = {"gt_ids": gt_ids, "boxes": [[round(x, 3) for x in b] for b in boxes],
                             "E0": [e0.get(di) for di in range(n)],
                             "E2": e2_ids,
                             "E3": e3_ids,
                             "E5": [e5.get(di) for di in range(n)],
                             "E7": e7_ids,
                             "E8": e8_ids}
    return out


def stage_track(args) -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    import generate_gt_tool_cases  # noqa: F401  (sets tracker sys.path)
    vids = select_videos(args)
    done, waiting = 0, 0
    for v in vids:
        out = CACHE / f"{v['key']}.tracks.json"
        if out.exists():
            continue
        sp, cp = CACHE / f"{v['key']}.sam.json", CACHE / f"{v['key']}.classify.json"
        if not (sp.exists() and cp.exists()):
            waiting += 1
            continue
        try:
            res = track_video(v, json.loads(sp.read_text(encoding="utf-8")),
                              json.loads(cp.read_text(encoding="utf-8"))["e8_medians"])
            atomic_write(out, res)
            done += 1
            log({"stage": "track", "video": v["video_id"], "ok": True, "frames": len(res)})
        except Exception as e:
            log({"stage": "track", "video": v["video_id"], "ok": False, "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
    print(f"[track] {done} videos tracked this run; {waiting} still waiting for sam/classify caches", flush=True)


# ---------------------------------------------------------------------------
# Stage: evaluate (writes the EXISTING condition_classification files in place)
# ---------------------------------------------------------------------------
IN_PLACE_FILES = [
    "all_cases_frame_conditions.csv", "frame_heuristic_association.csv", "summary_cell_counts.csv",
    "summary_cell_counts_by_case.csv", "heuristic_condition_summary.csv", "heuristic_summary_validation.json",
    "validation_report.json", "HEURISTIC_CONDITION_SUMMARY.md", "classification_summary.md",
    "threshold_configuration.json",
]
BACKUP_DIR = SCRIPT_DIR / "backups" / "pre_full_dataset_2026-10-13"


def backup_once() -> None:
    import shutil
    if BACKUP_DIR.exists():
        return
    BACKUP_DIR.mkdir(parents=True)
    for f in IN_PLACE_FILES:
        if (CC_DIR / f).exists():
            shutil.copy2(CC_DIR / f, BACKUP_DIR / f)
    shutil.copy2(SCRIPT_DIR / "README.md", BACKUP_DIR / "README.md")


def cluster_bootstrap(assoc, h: str, field: str, value: str, n_boot: int = 2000, seed: int = 0):
    """Video-level bootstrap of (rate_h - rate_E0) on the same frames; returns (diff_pp, lo, hi)."""
    per = defaultdict(lambda: [0, 0, 0])  # video -> [n, k_h, k_e0]
    idx = {(a["case_id"], a["frame_number"], a["heuristic"]): a["correct"] for a in assoc}
    for a in assoc:
        if a["heuristic"] != "E0" or not (field == "eligible" or a[field] == value):
            continue
        key = (a["case_id"], a["frame_number"], h)
        if key not in idx:
            continue
        p = per[a["case_id"]]
        p[0] += 1; p[1] += idx[key]; p[2] += a["correct"]
    if not per:
        return None
    arr = np.array(list(per.values()), dtype=float)
    diff = 100 * (arr[:, 1].sum() - arr[:, 2].sum()) / arr[:, 0].sum()
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = arr[rng.integers(0, len(arr), len(arr))]
        bs.append(100 * (s[:, 1].sum() - s[:, 2].sum()) / s[:, 0].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return round(float(diff), 2), round(float(lo), 2), round(float(hi), 2), len(arr)


def stage_evaluate(args) -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    import classify_frame_conditions as cfc
    import summarize_heuristic_conditions as shc

    all_vids = enumerate_videos(include_excluded=True)
    vids = [v for v in all_vids if v["schema"] == "reviewed"]
    excluded = [v["video_id"] for v in all_vids if v["schema"] != "reviewed"]
    rows: List[Dict[str, Any]] = []
    prechecks: List[Dict[str, Any]] = []
    assoc: List[Dict[str, Any]] = []
    validation: Dict[str, Any] = {"population": "full dataset", "videos_eligible_schema": len(vids),
                                  "videos_excluded_raw_tracker_schema": excluded,
                                  "videos_missing_cache": [], "box_match_failures": [], "frames_missing_in_tracks": []}
    same = {"E3==E7": True, "E0==E8": True}
    diff_frames = {"E3==E7": 0, "E0==E8": 0}
    for v in vids:
        cp, tp = CACHE / f"{v['key']}.classify.json", CACHE / f"{v['key']}.tracks.json"
        if not (cp.exists() and tp.exists()):
            validation["videos_missing_cache"].append(v["video_id"])
            continue
        c = json.loads(cp.read_text(encoding="utf-8"))
        tr = json.loads(tp.read_text(encoding="utf-8"))
        # Existing classification caches predate the 0.5 very-high boundary.
        # Split their former >0.25 HIGH_OVERLAP rows deterministically from the
        # cached GT-derived maximum IoU before any aggregation.
        for r in c["rows"]:
            if (r["eligible"] == 1 and r["overlap_label"] == "HIGH_OVERLAP"
                    and r["max_overlap_iou"] not in ("", None)
                    and float(r["max_overlap_iou"]) > 0.5):
                r["overlap_label"] = "VERY_HIGH_OVERLAP"
        rows.extend(c["rows"])
        prechecks.append(c["pre"])
        for fr in tr.values():
            for a, b in (("E3", "E7"), ("E0", "E8")):
                if fr[a] != fr[b]:
                    same[f"{a}=={b}"] = False
                    diff_frames[f"{a}=={b}"] += 1
        elig = sorted((r for r in c["rows"] if r["eligible"] == 1), key=lambda r: r["frame_number"])
        for h in shc.HEURISTICS:
            per_frame = []
            for r in elig:
                fr = tr.get(r["frame_name"])
                if fr is None:
                    validation["frames_missing_in_tracks"].append(f"{v['video_id']}/{r['frame_name']}")
                    continue
                target = [float(r[k]) for k in ("raider_x1", "raider_y1", "raider_x2", "raider_y2")]
                cand = [i for i, (b, g) in enumerate(zip(fr["boxes"], fr["gt_ids"]))
                        if g == int(r["raider_gt_id"]) and all(abs(x - y) <= 0.006 for x, y in zip(b, target))]
                if len(cand) != 1:
                    validation["box_match_failures"].append(f"{v['video_id']}/{h}/{r['frame_name']}")
                    continue
                ids = fr[h]
                per_frame.append({"row": {**r, "frame_number": r["frame_number"]}, "tid": ids[cand[0]],
                                  "ids": ids, "ridx": cand[0]})
            assoc.extend(shc.judge_frames(per_frame, v["video_id"], h, validation))
    validation["identical_outputs"] = same
    validation["frames_where_outputs_differ"] = diff_frames
    n_ph = len(validation.get("birth_placeholder_suspects", []))
    validation["birth_placeholder_suspects_count"] = n_ph

    table, cell_rows, by_video = shc.compute_tables(assoc)
    cfg = json.loads(json.dumps(cfc.DEFAULT_CONFIG))
    post = cfc.post_validate(rows, [v["video_id"] for v in vids if v["video_id"] not in validation["videos_missing_cache"]], cfg)
    boot = {h: {lab: cluster_bootstrap(assoc, h, f, val) for f, val, lab in shc.CONDITIONS + [shc.ALL]}
            for h in shc.HEURISTICS if h != "E0"}
    validation["e0_difference_cluster_bootstrap_pp"] = boot

    # diagnostic (selected cases) section, from the unchanged per-case files
    d_rows, d_assoc, d_val, d_ident = shc.build_diagnostic()
    d_table, _, _ = shc.compute_tables(d_assoc)

    out_dir = Path(args.dry_run) if args.dry_run else CC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        backup_once()
    rows_sorted = sorted(rows, key=lambda r: (r["case_id"], r["frame_number"]))
    with (out_dir / "all_cases_frame_conditions.csv").open("w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=cfc.CSV_FIELDS)
        w.writeheader()
        w.writerows(rows_sorted)
    shc.write_csv(out_dir / "frame_heuristic_association.csv", assoc)
    shc.write_csv(out_dir / "summary_cell_counts.csv", cell_rows)
    shc.write_csv(out_dir / "summary_cell_counts_by_case.csv", by_video)
    shc.write_summary_csv(out_dir / "heuristic_condition_summary.csv", table, same)
    (out_dir / "threshold_configuration.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "heuristic_summary_validation.json").write_text(
        json.dumps({"full_dataset": validation, "diagnostic_selected_cases": d_val}, indent=2), encoding="utf-8")
    (out_dir / "validation_report.json").write_text(
        json.dumps({"population": "full dataset", "post_run": post, "prechecks": prechecks}, indent=2), encoding="utf-8")

    import full_dataset_reports as rep
    ctx = dict(rows=rows, assoc=assoc, table=table, by_video=by_video, validation=validation, same=same,
               d_table=d_table, d_ident=d_ident, d_val=d_val, d_rows=d_rows, assoc_diag=d_assoc, prechecks=prechecks, post=post,
               cfg=cfg, boot=boot, vids=vids, excluded=excluded, all_vids=all_vids, shc=shc)
    (out_dir / "HEURISTIC_CONDITION_SUMMARY.md").write_text(rep.heuristic_md(ctx), encoding="utf-8")
    (out_dir / "classification_summary.md").write_text(rep.classification_md(ctx), encoding="utf-8")
    if not args.dry_run:
        rep.update_readme(ctx, SCRIPT_DIR / "README.md")
    print(json.dumps({k: v for k, v in validation.items() if k not in ("e0_difference_cluster_bootstrap_pp", "birth_placeholder_suspects")}, indent=1)[:3000])
    print("post_run:", post)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["inventory", "sam", "classify", "track", "evaluate"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--dry-run", metavar="DIR", help="evaluate: write outputs to DIR instead of in place")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.stage == "inventory":
        vids = select_videos(args)
        print(len(vids), "videos")
    else:
        globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
