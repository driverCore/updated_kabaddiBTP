"""
sam_depth_pipeline.py
─────────────────────
E7 supplement — Combined SAM + MiDaS depth pipeline.

Runs both SAM automatic mask generation (segment everything) and
MiDaS depth estimation on a video, computing per-mask median depth
and relative depth values.

This provides the mask-level depth features that can be used for
mask IoU association combined with depth gating.

Usage
-----
    python sam_depth_pipeline.py --video input.mp4
"""

import os
import time
import csv

import cv2
import numpy as np
import torch

# ── Configurable Paths ────────────────────────────────────────────────────

SAM_CHECKPOINT = os.path.join("models", "mobile_sam.pt")
OUTPUT_DIR = os.path.join("output", "sam_depth")


# ── MiDaS Loading ─────────────────────────────────────────────────────────

def load_midas():
    """Load MiDaS Small and its transforms exactly once."""
    print("[INFO] Loading MiDaS Small …")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform

    print(f"[INFO] MiDaS Small loaded on {device}")
    return model, transform, device


@torch.no_grad()
def compute_depth_map(frame_bgr, model, transform, device):
    """Return a float32 depth map (H, W) for a BGR frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(frame_rgb).to(device)

    prediction = model(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame_bgr.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    return prediction.cpu().numpy()


def depth_to_colormap(depth_map):
    """Normalise depth to 0-255 and apply a colour map for visualisation."""
    d = depth_map.copy()
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        d = (d - d_min) / (d_max - d_min)
    else:
        d = np.zeros_like(d)
    d = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)


# ── SAM Loading ───────────────────────────────────────────────────────────

def load_sam():
    """Load MobileSAM with automatic mask generator (segment everything)."""
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

    print("[INFO] Loading MobileSAM …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_t"](checkpoint=SAM_CHECKPOINT)
    sam.to(device).eval()

    mask_generator = SamAutomaticMaskGenerator(sam)
    print(f"[INFO] MobileSAM loaded on {device}")
    return mask_generator


def sam_visualise(frame_bgr, masks):
    """
    Overlay every SAM mask with a unique random colour on top of the frame.

    Parameters
    ----------
    frame_bgr : np.ndarray  (H, W, 3)
    masks     : list[dict]  – output of SamAutomaticMaskGenerator.generate()

    Returns
    -------
    np.ndarray (H, W, 3)  – visualisation image
    """
    overlay = frame_bgr.copy().astype(np.float32)
    np.random.seed(42)  # reproducible colours across frames
    colours = np.random.randint(50, 256, size=(
        max(len(masks), 1), 3), dtype=np.uint8)

    # Sort masks by area (largest first) so smaller masks paint on top
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    for idx, mask_data in enumerate(sorted_masks):
        mask = mask_data["segmentation"]  # bool (H, W)
        colour = colours[idx % len(colours)].astype(np.float32)
        overlay[mask] = overlay[mask] * 0.45 + colour * 0.55

    return overlay.astype(np.uint8)


# ── Main Pipeline ─────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM + MiDaS Depth Pipeline")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    args = parser.parse_args()

    video_path = args.video
    output_dir = args.output_dir
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    depth_frames_dir = os.path.join(output_dir, f"{video_name}_depth_frames")
    sam_frames_dir = os.path.join(output_dir, f"{video_name}_sam_frames")
    os.makedirs(depth_frames_dir, exist_ok=True)
    os.makedirs(sam_frames_dir, exist_ok=True)

    print(f"Input Video  : {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Total Frames : {total_frames}")
    print(f"Resolution   : {width}×{height} @ {fps:.1f} FPS")

    # Load models
    midas_model, midas_transform, midas_device = load_midas()
    mask_generator = load_sam()

    # Video writers
    depth_video_path = os.path.join(output_dir, f"{video_name}_depth.mp4")
    sam_video_path = os.path.join(output_dir, f"{video_name}_sam.mp4")
    csv_path = os.path.join(output_dir, f"{video_name}_sam_depth.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    depth_writer = cv2.VideoWriter(depth_video_path, fourcc, fps, (width, height))
    sam_writer = cv2.VideoWriter(sam_video_path, fourcc, fps, (width, height))

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "MaskIndex", "Area", "BBox_x",
                        "BBox_y", "BBox_w", "BBox_h", "MedianDepth", "RelativeDepth"])

    # Process frames
    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx} / {total_frames}")

        # 1. Depth
        depth_map = compute_depth_map(frame, midas_model, midas_transform, midas_device)
        depth_vis = depth_to_colormap(depth_map)

        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            norm_depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            norm_depth_map = np.zeros_like(depth_map)

        depth_writer.write(depth_vis)

        # 2. SAM (segment everything)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(frame_rgb)

        sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)

        for m_idx, mask_data in enumerate(sorted_masks):
            mask_bool = mask_data["segmentation"]
            bbox = mask_data["bbox"]
            area = mask_data["area"]

            if np.any(mask_bool):
                med_depth = float(np.median(depth_map[mask_bool]))
                rel_depth = float(np.median(norm_depth_map[mask_bool]))
            else:
                med_depth = 0.0
                rel_depth = 0.0

            csv_writer.writerow([
                frame_idx, m_idx, area,
                round(bbox[0], 2), round(bbox[1], 2),
                round(bbox[2], 2), round(bbox[3], 2),
                round(med_depth, 4), round(rel_depth, 4)
            ])

        sam_vis = sam_visualise(frame, masks)
        sam_writer.write(sam_vis)

        frame_idx += 1

    cap.release()
    depth_writer.release()
    sam_writer.release()
    csv_file.close()

    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_idx} frames in {elapsed:.2f} seconds")
    print(f"Depth video : {depth_video_path}")
    print(f"SAM video   : {sam_video_path}")
    print(f"CSV log     : {csv_path}")


if __name__ == "__main__":
    main()
