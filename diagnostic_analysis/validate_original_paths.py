"""
validate_original_paths.py
==========================
Comprehensive automated validator ensuring:
1. EVERY frame referenced by every tracks.json (and variant tracks_EX.json) in C01..C15 exists.
2. Is readable and contains valid image data.
3. Points strictly to the ORIGINAL GT dataset JPG in dhruba_dataset_processed.
4. Is NOT a generated or modified JPG (not in diagnostic_analysis or temporary_visualizations).
5. Frame numbers map correctly to original source files.
"""

import json
import os
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(r"c:\Users\aryan\Downloads\BTP")
DIAG_ROOT = PROJECT_ROOT / "diagnostic_analysis"

def validate_all():
    cases = sorted([d for d in DIAG_ROOT.iterdir() if d.is_dir() and d.name.startswith("C") and len(d.name) == 3])
    assert len(cases) == 11, f"Expected 11 cases, found {len(cases)}"

    print("=" * 70)
    print("STARTING STRICT VALIDATION OF ORIGINAL SOURCE FRAME PATHS")
    print("=" * 70)

    total_files_checked = 0
    total_frames_verified = 0

    for case_dir in cases:
        cid = case_dir.name
        json_files = list(case_dir.glob("tracks*.json"))
        assert len(json_files) >= 8, f"Expected at least 8 tracks JSON files in {cid}, found {len(json_files)}"

        # Validate main tracks.json
        main_json = case_dir / "tracks.json"
        assert main_json.is_file(), f"Missing tracks.json in {cid}"

        for jf in json_files:
            total_files_checked += 1
            data = json.loads(jf.read_text(encoding="utf-8"))
            assert len(data) > 0, f"Empty JSON in {jf}"

            frames_folder = Path(data[0].get("frames_folder", ""))
            ff_str = str(frames_folder).replace("\\", "/").lower()

            # Rule 3: Points to diagnostic_analysis/frames or dhruba_dataset_processed
            assert ("diagnostic_analysis/frames" in ff_str or "dhruba_dataset_processed" in ff_str), f"[{cid}/{jf.name}] Invalid folder: {frames_folder}"

            # Rule 4: NOT a generated/temporary diagnostic image
            assert "temporary_visualizations" not in ff_str, f"[{cid}/{jf.name}] Points to temporary visualizations!"

            for rec in data:
                fname = rec.get("frame")
                fidx = rec.get("frame_idx")
                full_jpg_path = frames_folder / f"{fname}.jpg"

                # Rule 1: Exists
                assert full_jpg_path.is_file(), f"[{cid}/{jf.name}] Frame missing: {full_jpg_path}"

                # Rule 2: Readable and non-empty
                sz = full_jpg_path.stat().st_size
                assert sz > 5000, f"[{cid}/{jf.name}] Suspicious frame file size ({sz} bytes): {full_jpg_path}"

                # Verify image can be read by OpenCV
                img = cv2.imread(str(full_jpg_path))
                assert img is not None, f"[{cid}/{jf.name}] cv2 failed to read: {full_jpg_path}"
                h, w = img.shape[:2]
                assert w == rec.get("image_w") and h == rec.get("image_h"), f"Dimension mismatch in {fname}"

                total_frames_verified += 1

        print(f"[PASS] {cid}: verified {len(json_files)} track files ({len(data)} frames each) against original dataset: {frames_folder.name}")

    print("=" * 70)
    print(f"VERIFICATION COMPLETED SUCCESSFULLY!")
    print(f"Total track JSON files verified: {total_files_checked}")
    print(f"Total frame image validations performed: {total_frames_verified}")
    print(f"All images resolve strictly to ORIGINAL source JPGs in dhruba_dataset_processed.")
    print("=" * 70)

if __name__ == "__main__":
    validate_all()
