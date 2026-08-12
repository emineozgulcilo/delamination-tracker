#!/usr/bin/env python3
"""
main.py  —  Delamination Crack Propagation Tracker
===================================================

Usage examples
--------------
  # Basic run: select ROI interactively, use pixel units
  python main.py path/to/video.mp4

  # With calibration (pixels per mm known)
  python main.py video.mp4 --pixels-per-mm 12.5

  # Click two calibration points interactively
  python main.py video.mp4 --calibrate

  # Left-propagating crack, no live display (batch mode)
  python main.py video.mp4 --direction left --no-display

  # Custom output directory and min gap
  python main.py video.mp4 --output-dir results/ --min-gap 8

Keyboard shortcuts during live display
---------------------------------------
  Q / ESC  — stop early
  SPACE    — pause / resume
  S        — save a snapshot of the current overlay frame
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# All modules use relative imports because main.py is inside the package dir.
# Running with  `python tracker/main.py`  or  `python -m tracker.main`  both work.
try:
    from video_reader import VideoReader
    from roi import ROI, ROISelector
    from edge_detection import EdgeDetector
    from crack_tracker import CrackTracker
    from measurement import Calibrator, MeasurementStore
    from visualization import Visualizer
    from export import Exporter
except ModuleNotFoundError:
    # Fallback for `python -m tracker.main` style invocation
    from tracker.video_reader import VideoReader
    from tracker.roi import ROI, ROISelector
    from tracker.edge_detection import EdgeDetector
    from tracker.crack_tracker import CrackTracker
    from tracker.measurement import Calibrator, MeasurementStore
    from tracker.visualization import Visualizer
    from tracker.export import Exporter


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delamination Crack Propagation Tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="Path to the input video file")
    p.add_argument("--output-dir", default="output",
                   help="Directory for all output files")
    p.add_argument("--pixels-per-mm", type=float, default=None,
                   help="Calibration constant.  Omit to work in pixels.")
    p.add_argument("--calibrate", action="store_true",
                   help="Interactively pick two points with a known distance")
    p.add_argument("--min-gap", type=int, default=3,
                   help="Minimum crack-gap height in pixels")
    p.add_argument("--direction", choices=["right", "left"], default="right",
                   help="Direction in which the crack propagates")
    p.add_argument("--no-display", action="store_true",
                   help="Suppress the live display window (faster batch mode)")
    p.add_argument("--smooth-window", type=int, default=7,
                   help="Rolling-median window size for tip smoothing (frames)")
    p.add_argument("--clahe-clip", type=float, default=1.5,
                   help="CLAHE clip limit for contrast enhancement")
    p.add_argument("--bright-crack", action="store_true",
                   help="Crack gap is brighter than the arms (reflective fracture surface). "
                        "Default: gap is darker (shadow in open crack).")
    p.add_argument("--skip-frames", type=int, default=0,
                   help="Process every Nth frame only (0 = all frames)")
    p.add_argument("--roi", default=None,
                   help="Fixed ROI as x,y,w,h (skips interactive selection)")
    p.add_argument("--initial-crack-length", type=float, default=0.0,
                   help="Initial pre-crack length a0 in mm (added to measured propagation). "
                        "Example: --initial-crack-length 50.0")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------

def run_calibration(
    args: argparse.Namespace,
    first_frame: np.ndarray,
) -> Calibrator:
    cal = Calibrator(pixels_per_mm=args.pixels_per_mm or 1.0)

    if args.calibrate:
        result = cal.select_points(first_frame)
        if result is None:
            print("[Calibration] Skipped — using pixel units.")
    elif args.pixels_per_mm is not None:
        print(f"[Calibration] Using {args.pixels_per_mm} px/mm from command line.")
    else:
        resp = input(
            "\nEnter pixels per mm (or press ENTER to work in pixel units): "
        ).strip()
        if resp:
            try:
                cal.pixels_per_mm = float(resp)
                print(f"[Calibration] Using {cal.pixels_per_mm} px/mm.")
            except ValueError:
                print("[Calibration] Invalid — using pixel units.")

    return cal


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------

def process(
    reader: VideoReader,
    roi: ROI,
    detector: EdgeDetector,
    tracker: CrackTracker,
    store: MeasurementStore,
    visualizer: Visualizer,
    exporter: Exporter,
    args: argparse.Namespace,
):
    video_writer = exporter.create_video_writer(
        "annotated_video.mp4",
        fps=reader.fps,
        width=reader.width,
        height=reader.height,
    )

    display_name = "Delamination Tracker  [Q/ESC=quit  SPACE=pause  S=snapshot]"
    if not args.no_display:
        cv2.namedWindow(display_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(display_name, min(reader.width, 1280),
                         min(reader.height, 720))

    reader.reset()
    frame_idx = 0
    paused = False
    snapshot_count = 0
    t_start = time.time()

    print(f"\n[Processing] {reader.frame_count} frames  |  "
          f"{reader.fps:.2f} fps  |  "
          f"ROI {roi.w}×{roi.h} px\n")

    while True:
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = False
            continue

        ret, frame = reader.read_frame()
        if not ret:
            break

        # Optional frame skipping for speed
        step = max(1, args.skip_frames + 1)
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # ── Detection ──────────────────────────────────────────────────
        roi_crop = frame[roi.slice_yx]
        upper_local, lower_local = detector.detect(roi_crop)

        # ── Tracking ───────────────────────────────────────────────────
        state = tracker.update(
            frame_idx, upper_local, lower_local,
            subpixel_tip_x=detector._subpixel_tip_x,
        )

        # ── Convert to global coords for visualisation ─────────────────
        upper_global = roi.pts_to_global(upper_local)
        lower_global = roi.pts_to_global(lower_local)
        tip_global   = roi.to_global(state.tip_x, state.tip_y)
        init_global  = (roi.to_global(*tracker.initial_tip)
                        if tracker.initial_tip else None)

        # ── Measurement ────────────────────────────────────────────────
        m = store.record(
            frame_index=frame_idx,
            crack_length_px=state.propagation_h_px,
            tip_global=tip_global,
            gap_at_tip_px=state.gap_at_tip,
        )

        # ── Visualisation ──────────────────────────────────────────────
        out_frame = visualizer.draw_overlay(
            frame=frame,
            upper_pts_global=upper_global,
            lower_pts_global=lower_global,
            tip_global=tip_global,
            initial_tip_global=init_global,
            measurement=m,
            fps=reader.fps,
            roi_rect=(roi.x, roi.y, roi.w, roi.h),
        )

        video_writer.write(out_frame)

        # ── Live display ───────────────────────────────────────────────
        if not args.no_display:
            cv2.imshow(display_name, out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\n[Processing] Stopped by user.")
                break
            if key == ord(" "):
                paused = True
            if key == ord("s"):
                snap_path = Path(args.output_dir) / f"snapshot_{snapshot_count:04d}.png"
                cv2.imwrite(str(snap_path), out_frame)
                snapshot_count += 1
                print(f"[Snapshot] {snap_path}")

        # ── Progress ───────────────────────────────────────────────────
        if frame_idx % 30 == 0:
            elapsed = time.time() - t_start
            pct = (frame_idx / reader.frame_count * 100) if reader.frame_count else 0
            print(
                f"  Frame {frame_idx:>6d} / {reader.frame_count}"
                f" ({pct:5.1f}%)"
                f"  tip=({tip_global[0]:4d},{tip_global[1]:4d})"
                f"  length={m.crack_length_mm:8.3f} mm"
                f"  elapsed={elapsed:.1f}s"
            )

        frame_idx += 1

    # ── Cleanup ────────────────────────────────────────────────────────
    video_writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    total = time.time() - t_start
    print(f"\n[Processing] Done — {frame_idx} frames in {total:.1f}s "
          f"({frame_idx / total:.1f} fps effective)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Open video ─────────────────────────────────────────────────────
    reader = VideoReader(args.video)
    print(f"\n[Video] {reader}")

    first_frame = reader.get_first_frame()

    # ── ROI selection ──────────────────────────────────────────────────
    if args.roi:
        try:
            rx, ry, rw, rh = (int(v) for v in args.roi.split(","))
            roi = ROI(x=rx, y=ry, w=rw, h=rh)
            print(f"[ROI] Using fixed ROI: x={roi.x}, y={roi.y}, "
                  f"w={roi.w}, h={roi.h}")
        except (ValueError, TypeError):
            print("[ROI] Invalid --roi format — expected x,y,w,h. Using full frame.")
            roi = ROI(x=0, y=0, w=reader.width, h=reader.height)
    else:
        selector = ROISelector()
        print("\n[ROI] Draw a rectangle around the crack region.")
        try:
            roi = selector.select(first_frame)
            print(f"[ROI] Selected: x={roi.x}, y={roi.y}, "
                  f"w={roi.w}, h={roi.h}")
        except ValueError:
            roi = ROI(x=0, y=0, w=reader.width, h=reader.height)
            print("[ROI] No selection — using full frame.")

    # ── Calibration ────────────────────────────────────────────────────
    calibrator = run_calibration(args, first_frame)

    # ── Module initialisation ──────────────────────────────────────────
    detector = EdgeDetector(
        min_gap_pixels=args.min_gap,
        clahe_clip=1.5,
        min_arm_brightness=145.0,
        min_lower_brightness=130.0,
        max_gap_search=65,
        max_gap_pixels=90,
        max_lower_y_frac=0.82,
        bright_crack=args.bright_crack,
        mode="threshold",
    )
    tracker = CrackTracker(
        min_gap_pixels=1,
        smooth_window=args.smooth_window,
        crack_direction=args.direction,
    )
    store      = MeasurementStore(fps=reader.fps, calibrator=calibrator,
                                  initial_crack_mm=args.initial_crack_length)
    visualizer = Visualizer()
    exporter   = Exporter(output_dir=args.output_dir)

    # ── Main loop ──────────────────────────────────────────────────────
    process(reader, roi, detector, tracker, store, visualizer, exporter, args)
    reader.release()

    # ── Export ─────────────────────────────────────────────────────────
    if store.records:
        exporter.export_csv(store.records)
        exporter.export_plot(store.records)
        exporter.print_summary(store.records, reader.fps)
    else:
        print("[Export] No measurements collected.")

    print(f"\n[Done] Outputs in: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
