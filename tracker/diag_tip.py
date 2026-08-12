#!/usr/bin/env python3
"""
diag_tip.py - Quick diagnostic: print crack tip x per frame on real video.
Run from the tracker/ directory:
  python diag_tip.py
"""
import sys, os, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from video_reader import VideoReader
from roi import ROI
from edge_detection import EdgeDetector
from crack_tracker import CrackTracker
from measurement import Calibrator, MeasurementStore

VIDEO = r"C:\Users\Süleyman\Desktop\dcb_test.mp4"
# Try to locate the video robustly
import glob as _glob
_candidates = _glob.glob(os.path.join(os.environ.get("USERPROFILE","C:\\Users\\*"), "Desktop", "dcb_test.mp4"))
if _candidates:
    VIDEO = _candidates[0]

ROI_RECT = ROI(x=100, y=190, w=470, h=150)
PX_PER_MM = 4.4

PROBE_FRAMES = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700,
                800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

def main():
    print(f"Video: {VIDEO}")
    reader = VideoReader(VIDEO)
    print(f"  {reader.frame_count} frames  {reader.fps:.1f} fps  "
          f"{reader.width}x{reader.height}")

    det = EdgeDetector(
        min_gap_pixels=5,
        min_run_pixels=12,
        min_threshold=110,
        min_arm_brightness=145.0,
        min_lower_brightness=130.0,
        max_gap_search=65,
        mode="threshold",
    )
    trk = CrackTracker(min_gap_pixels=5, smooth_window=7, crack_direction="right")
    cal = Calibrator(pixels_per_mm=PX_PER_MM)
    store = MeasurementStore(fps=reader.fps, calibrator=cal)

    probe_set = set(PROBE_FRAMES)
    reader.reset()
    print(f"\n{'Frame':>6}  {'Time(s)':>7}  {'TipX(global)':>13}  "
          f"{'TipX(local)':>11}  {'Gap(px)':>7}  {'Length(mm)':>10}")
    print("-" * 65)

    for i in range(reader.frame_count):
        ret, frame = reader.read_frame()
        if not ret:
            break
        crop = frame[ROI_RECT.slice_yx]
        upper_l, lower_l = det.detect(crop)
        state = trk.update(i, upper_l, lower_l)
        tip_g = ROI_RECT.to_global(state.tip_x, state.tip_y)
        m = store.record(i, state.propagation_px, tip_g, state.gap_at_tip)

        if i in probe_set:
            print(f"{i:>6}  {m.time_s:>7.2f}  {tip_g[0]:>13}  "
                  f"{state.tip_x:>11}  {state.gap_at_tip:>7}  "
                  f"{m.crack_length_mm:>10.3f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
