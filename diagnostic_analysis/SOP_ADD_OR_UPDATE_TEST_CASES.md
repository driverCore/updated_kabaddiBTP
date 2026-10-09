# Standard Operating Procedure (SOP): Adding, Replacing, or Re-Evaluating Test Cases

**Target**: LLM Coding Agents & Developers  
**Repository**: `c:\Users\aryan\Downloads\BTP`  
**Purpose**: Complete, step-by-step instruction manual on how to process, evaluate, generate, validate, and document new or updated diagnostic test cases across the benchmark suite.

---

## Overview of What Happens During a Test Case Update

Whenever the user asks to:
- **Re-run / re-evaluate an existing case** (e.g., changes made to original `tracks.json`),
- **Replace old cases** with new sequences (e.g., replace C05/C06 with new actions), or
- **Add new cases in sequence** (e.g., extend C09 with C10, C11, etc.),

you must execute a **7-step pipeline** spanning sequence discovery, heuristic execution, visual rendering, automated verification, and updating all 7 documentation files.

---

## Step 1: Discover & Inspect the Original Video Sequence

All source datasets live under `dhruba_dataset_processed/`:
- Raid actions: `dhruba_dataset_processed/raid/<action>/bytetrack/bytetrack_<seq>_frames/`
- Defend actions: `dhruba_dataset_processed/defend/<action>/bytetrack/bytetrack_<seq>_frames/`

### Actions to Perform:
1. Locate the sequence's original `tracks.json`.
2. Inspect the frames to identify:
   - Total frame count and dimensions (width, height).
   - Raider ID (e.g., who is raiding) and defender IDs.
   - The **critical event frame** (`crit_frame`): the exact frame where maximum overlap, collision, contact, or ducking occurs.
   - The **evaluation window**: select a window of ~15 to 45 frames around the critical frame (e.g., `start_frame` to `end_frame`).

---

## Step 2: Cleanup (Only When Replacing or Removing Cases)

If replacing or removing an old case ID (e.g., removing old C05 or C06):
1. Delete the case directory: `Remove-Item -Recurse -Force diagnostic_analysis/<CASE_ID>`
2. Delete the global frame directory: `Remove-Item -Recurse -Force diagnostic_analysis/frames/<CASE_ID>`

---

## Step 3: Update `diagnostic_analysis/generate_gt_tool_cases.py`

This script is the **single source of truth** that generates all tracking files, runs heuristic simulations, and renders visual frames.

### Actions to Perform:
1. Locate the `CASES` dictionary in `diagnostic_analysis/generate_gt_tool_cases.py`.
2. Add or update the case entry:
   ```python
   "C10": {
       "seq": "wh41",
       "action": "waist_hold",
       "split": "defend",
       "crit_frame": 90,
       "start_frame": 76,
       "end_frame": 90,
       "phenomenon": "Multi-defender waist hold pack capture with severe occlusion clutter (peak IoU 0.375 at frame 90)",
   },
   ```
3. Run the generation script:
   ```powershell
   python diagnostic_analysis/generate_gt_tool_cases.py
   ```
4. **What this script automatically produces**:
   - For every case:
     - `diagnostic_analysis/<CASE_ID>/tracks.json` (active track file, initially set to E0)
     - `diagnostic_analysis/<CASE_ID>/tracks_E0.json` (Baseline ByteTrack)
     - `diagnostic_analysis/<CASE_ID>/tracks_E2.json` (Torso ROI Slicing)
     - `diagnostic_analysis/<CASE_ID>/tracks_E3.json` (Pose Keypoints Alignment)
     - `diagnostic_analysis/<CASE_ID>/tracks_E5.json` (Hip/Shorts HSV Colour)
     - `diagnostic_analysis/<CASE_ID>/tracks_E7.json` (SAM Mask Silhouette IoU)
     - `diagnostic_analysis/<CASE_ID>/tracks_E8.json` (MiDaS Relative Depth)
     - `diagnostic_analysis/<CASE_ID>/tracks_GT.json` (Verified Ground Truth)
     - `diagnostic_analysis/<CASE_ID>/frames/` & `frames_EX/` (annotated frames with GREEN=raider, RED=defender, and top HUD banner showing active heuristic and switch count)
     - `diagnostic_analysis/frames/<CASE_ID>/` (global frame store used by the GT Editor)
   - Benchmark summary:
     - `diagnostic_analysis/experiment_results_summary.json` (machine-readable metrics with switch counts, S/P/F statuses, and totals)

---

## Step 4: Update & Run Automated Verification Scripts

### 1. Dataset & Path Integrity Validator (`diagnostic_analysis/validate_original_paths.py`):
- Open `diagnostic_analysis/validate_original_paths.py`.
- Update the assertion line to match the new total case count:
  ```python
  assert len(cases) == <NEW_TOTAL_CASES>, f"Expected <NEW_TOTAL_CASES> cases, found {len(cases)}"
  ```
- Run the validator:
  ```powershell
  python diagnostic_analysis/validate_original_paths.py
  ```
- Must verify: 100% of track JSON files exist, frames point to original images, images are non-empty and readable via OpenCV, and dimensions match.

### 2. Global Heuristic Switcher (`diagnostic_analysis/select_heuristic.py`):
- Test switching heuristics globally:
  ```powershell
  python diagnostic_analysis/select_heuristic.py E0
  ```
- Confirms that all cases sync their active `tracks.json` and frame images.

### 3. In-Application GT Tool Test Suite (`test_gt_tool_cases.py`):
- Open `test_gt_tool_cases.py`.
- Add the new case IDs to `cases_to_test`.
- Run the headless GUI test:
  ```powershell
  python test_gt_tool_cases.py
  ```
- Must confirm: PySide6 canvas loads images, tracks parse properly, slider navigates to critical frame, and step forward/backward works.

---

## Step 5: Read the Generated Metrics from `experiment_results_summary.json`

Open `diagnostic_analysis/experiment_results_summary.json` to get the exact:
1. `max_iou`: Peak bounding-box IoU between raider and defender at the critical frame.
2. `switches`: Exact number of raider ID switches for E0, E2, E3, E5, E7, E8.
3. `status`:
   - **S (Success)**: 0 switches.
   - **P (Partial)**: Exactly 1 transient switch before recovery.
   - **F (Failure)**: $\ge 2$ switches or permanent identity transfer.
4. `totals`: Updated global switch counts, total S, total P, total F across all heuristics.

---

## Step 6: Update ALL 7 Documentation Records (MANDATORY)

You must update all 7 documentation files to ensure complete consistency:

| File | What to Update |
|---|---|
| 1. [`CASE_SELECTION.md`](file:///c:/Users/aryan/Downloads/BTP/diagnostic_analysis/CASE_SELECTION.md) | Update total case count. Add/update table row (Sequence, Action, Frame Range, Critical Frame, Max IoU, Raider ID, Phenomenon). Add detailed per-case physical challenge profile. |
| 2. [`CASE_MATRIX.md`](file:///c:/Users/aryan/Downloads/BTP/diagnostic_analysis/CASE_MATRIX.md) | Update Section 1 evaluation matrix table (S / P / F status for all cases). Update Section 2 quantitative summary table (Successes, Partials, Failures, Total Switches). Update Section 3 high-level takeaways. |
| 3. [`OBSERVATIONS.md`](file:///c:/Users/aryan/Downloads/BTP/diagnostic_analysis/OBSERVATIONS.md) | Add/update the per-case technical observation entry with exact frame, action, difficulty, and cue-by-cue behavioral explanation for E0, E2, E3, E5, E7, E8. |
| 4. [`HEURISTIC_SUMMARY.md`](file:///c:/Users/aryan/Downloads/BTP/diagnostic_analysis/HEURISTIC_SUMMARY.md) | Update case count scope. Update quantitative performance matrix. Synthesize qualitative strengths and weaknesses for each heuristic incorporating the new cases. |
| 5. [`PLAN.md`](file:///c:/Users/aryan/Downloads/BTP/diagnostic_analysis/PLAN.md) | Update Section 3 case table and checklist to reflect the new cases and statuses. |
| 6. [`README.md`](file:///c:/Users/aryan/Downloads/BTP/diagnostic_analysis/README.md) | Update directory tree, summary table, CLI examples, and quantitative comparison section. |
| 7. [`CONTEXT_HANDOVER.md`](file:///c:/Users/aryan/Downloads/BTP/diagnostic_analysis/CONTEXT_HANDOVER.md) | Update the handover document so subsequent agents immediately have the latest state. |

---

## Step 7: Final User Reporting & Inspection Commands

In your response to the user:
1. Present the updated **Case Matrix Table** with S/P/F and switch counts.
2. Present the updated **Quantitative Summary Table** comparing all 6 heuristics.
3. Provide the key **qualitative takeaways** (why the new cases succeeded or failed).
4. Provide the exact commands to launch the GT Editor and inspect the new cases:
   ```powershell
   python ground_truth/editor_app/main.py diagnostic_analysis/<CASE_ID>/tracks.json
   ```
