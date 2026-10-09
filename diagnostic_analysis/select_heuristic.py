"""
select_heuristic.py
===================
Utility to switch the active tracks.json AND annotated frame JPGs across cases C02, C03, C07, C10
to any heuristic or GT baseline for seamless inspection in the GT Tool or external viewers.

Usage:
    python select_heuristic.py E0    # Set all cases to ByteTrack E0 baseline
    python select_heuristic.py E2    # Set all cases to Torso ROI E2
    python select_heuristic.py E3    # Set all cases to Pose Alignment E3
    python select_heuristic.py E5    # Set all cases to Hip Colour E5
    python select_heuristic.py E7    # Set all cases to SAM Mask IoU E7
    python select_heuristic.py E8    # Set all cases to Depth ByteTrack E8
    python select_heuristic.py GT    # Set all cases to Ground Truth reference
"""

import sys
import shutil
from pathlib import Path

VALID_HEURISTICS = ["E0", "E2", "E3", "E5", "E7", "E8", "GT"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python select_heuristic.py <E0|E2|E3|E5|E7|E8|GT>")
        print(f"Available options: {', '.join(VALID_HEURISTICS)}")
        sys.exit(1)

    choice = sys.argv[1].upper()
    if choice not in VALID_HEURISTICS:
        print(f"Error: Unknown heuristic '{choice}'. Must be one of {VALID_HEURISTICS}")
        sys.exit(1)

    diag_dir = Path(__file__).resolve().parent
    cases = sorted([d for d in diag_dir.iterdir() if d.is_dir() and d.name.startswith("C") and len(d.name) == 3])

    if not cases:
        print(f"No cases (C01..C10) found under {diag_dir}")
        sys.exit(1)

    src_name = f"tracks_{choice}.json"
    updated = 0
    frames_synced = 0

    for case_dir in cases:
        src_file = case_dir / src_name
        dest_file = case_dir / "tracks.json"

        if not src_file.is_file():
            print(f"Warning: {src_file} does not exist!")
            continue

        shutil.copyfile(src_file, dest_file)

        # Sync annotated frame images for this heuristic
        src_frames_dir = case_dir / f"frames_{choice}"
        dst_frames_local = case_dir / "frames"
        dst_frames_global = diag_dir / "frames" / case_dir.name

        if src_frames_dir.is_dir():
            dst_frames_local.mkdir(parents=True, exist_ok=True)
            dst_frames_global.mkdir(parents=True, exist_ok=True)
            for jf in src_frames_dir.glob("*.jpg"):
                shutil.copyfile(jf, dst_frames_local / jf.name)
                shutil.copyfile(jf, dst_frames_global / jf.name)
                frames_synced += 1

        updated += 1

    print(f"[SUCCESS] Successfully switched tracks.json AND synced {frames_synced} annotated frame images to [{choice}] across {updated} cases in {diag_dir.name}!")
    print(f"You can now launch the GT Tool directly:")
    print(f"  python ground_truth/editor_app/main.py diagnostic_analysis/C02/tracks.json")

if __name__ == "__main__":
    main()
