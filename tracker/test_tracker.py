#!/usr/bin/env python3
"""
test_tracker.py
---------------
Generates a synthetic DCB-style delamination video and runs the full
tracker pipeline without any interactive prompts.

Produces:
  test_output/annotated_video.mp4
  test_output/measurements.csv
  test_output/length_vs_time.png
"""

import sys
import os
import cv2
import numpy as np
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from video_reader import VideoReader
from roi import ROI
from edge_detection import EdgeDetector
from crack_tracker import CrackTracker
from measurement import Calibrator, MeasurementStore
from visualization import Visualizer
from export import Exporter


# ── Synthetic video parameters ────────────────────────────────────────────────
W, H       = 640, 240   # frame size
FPS        = 30.0
N_FRAMES   = 90         # 3 seconds
CRACK_X0   = 60         # initial crack tip x
CRACK_DX   = 3.5        # pixels per frame propagation
ARM_H      = 70         # arm thickness (pixels)
GAP_H      = 22         # crack gap height (pixels)
PX_PER_MM  = 5.0        # calibration


def make_synthetic_video(path: str):
    """
    Render a DCB specimen opening:
    - Left of tip: two bright arms + dark gap
    - Right of tip: intact bonded specimen
    - Gaussian noise + slight vertical camera shake added every frame
    """
    rng = np.random.default_rng(42)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                              FPS, (W, H))
    cy = H // 2   # specimen centre line

    for f in range(N_FRAMES):
        tip_x = int(CRACK_X0 + f * CRACK_DX)
        shake = int(rng.integers(-2, 3))   # ±2 px vertical shake

        frame = np.full((H, W), 60, dtype=np.uint8)   # dark background

        # ── Intact region (right of tip) ──────────────────────────────
        specimen_top = cy - (ARM_H + GAP_H // 2) + shake
        specimen_bot = cy + (ARM_H + GAP_H // 2) + shake
        frame[specimen_top:specimen_bot, tip_x:] = 200

        # ── Cracked region (left of tip): upper arm ───────────────────
        upper_top = cy - (ARM_H + GAP_H // 2) + shake
        upper_bot = cy - GAP_H // 2 + shake
        frame[upper_top:upper_bot, :tip_x] = 215

        # ── Cracked region: lower arm ─────────────────────────────────
        lower_top = cy + GAP_H // 2 + shake
        lower_bot = cy + (ARM_H + GAP_H // 2) + shake
        frame[lower_top:lower_bot, :tip_x] = 215

        # ── Crack gap (very dark) ─────────────────────────────────────
        gap_top = cy - GAP_H // 2 + shake
        gap_bot = cy + GAP_H // 2 + shake
        frame[gap_top:gap_bot, :tip_x] = 25

        # ── Noise ─────────────────────────────────────────────────────
        noise = rng.integers(-12, 13, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    writer.release()
    print(f"[Synth] Video written: {path}  ({N_FRAMES} frames, {N_FRAMES/FPS:.1f}s)")


def run_tracker(video_path: str, output_dir: str):
    print(f"\n[Tracker] Processing: {video_path}")

    reader   = VideoReader(video_path)
    first    = reader.get_first_frame()

    # Fixed ROI: tight around the specimen (skip top/bottom background)
    cy = H // 2
    margin = ARM_H + GAP_H // 2 + 10
    roi = ROI(x=0, y=max(0, cy - margin), w=W, h=min(H, 2 * margin))
    print(f"[Tracker] ROI: x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}")

    calibrator = Calibrator(pixels_per_mm=PX_PER_MM)
    detector   = EdgeDetector(min_gap_pixels=8, clahe_clip=2.0, dilate_iters=3)
    tracker    = CrackTracker(min_gap_pixels=8, smooth_window=7, crack_direction="right")
    store      = MeasurementStore(fps=reader.fps, calibrator=calibrator)
    vis        = Visualizer()
    exporter   = Exporter(output_dir)
    writer     = exporter.create_video_writer("annotated_video.mp4",
                                               reader.fps, reader.width, reader.height)

    reader.reset()
    for i in range(reader.frame_count):
        ret, frame = reader.read_frame()
        if not ret:
            break

        crop              = frame[roi.slice_yx]
        upper_l, lower_l  = detector.detect(crop)
        state             = tracker.update(i, upper_l, lower_l)
        tip_g             = roi.to_global(state.tip_x, state.tip_y)
        init_g            = roi.to_global(*tracker.initial_tip) if tracker.initial_tip else None
        upper_g           = roi.pts_to_global(upper_l)
        lower_g           = roi.pts_to_global(lower_l)
        m                 = store.record(i, state.propagation_px, tip_g, state.gap_at_tip)

        out_frame = vis.draw_overlay(
            frame, upper_g, lower_g, tip_g, init_g, m, reader.fps,
            (roi.x, roi.y, roi.w, roi.h)
        )
        writer.write(out_frame)

        if i % 15 == 0:
            gt_tip = int(CRACK_X0 + i * CRACK_DX)
            print(f"  frame {i:3d}  tip=({tip_g[0]:3d},{tip_g[1]:3d})"
                  f"  gt_x={gt_tip:3d}"
                  f"  length={m.crack_length_mm:6.2f} mm"
                  f"  gap={state.gap_at_tip:2d}px")

    writer.release()
    reader.release()

    exporter.export_csv(store.records)
    exporter.export_plot(store.records)
    exporter.print_summary(store.records, FPS)

    # ── Accuracy report ───────────────────────────────────────────────────────
    records = store.records
    final_gt_mm = (CRACK_DX * (N_FRAMES - 1)) / PX_PER_MM
    final_mm    = records[-1].crack_length_mm if records else 0.0
    error_mm    = abs(final_mm - final_gt_mm)

    print(f"\n[Accuracy]")
    print(f"  Ground truth final length : {final_gt_mm:.2f} mm")
    print(f"  Measured final length     : {final_mm:.2f} mm")
    print(f"  Absolute error            : {error_mm:.2f} mm")

    return records


def main():
    out_dir = Path(__file__).parent / "test_output"
    out_dir.mkdir(exist_ok=True)
    video_path = str(out_dir / "synthetic_dcb.mp4")

    print("=" * 60)
    print("  DELAMINATION TRACKER — SYNTHETIC TEST")
    print("=" * 60)

    make_synthetic_video(video_path)
    records = run_tracker(video_path, str(out_dir))

    print(f"\n[Done] Outputs in: {out_dir.resolve()}")
    print(f"  - annotated_video.mp4")
    print(f"  - measurements.csv")
    print(f"  - length_vs_time.png")
    print(f"  - synthetic_dcb.mp4  (input)")


if __name__ == "__main__":
    main()
