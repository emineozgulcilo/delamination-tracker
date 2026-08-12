#!/usr/bin/env python3
"""
diag_frames.py  — Save annotated frames at key points to verify detection.
"""
import sys, os, cv2, numpy as np, glob as _glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from video_reader import VideoReader
from roi import ROI
from edge_detection import EdgeDetector
from crack_tracker import CrackTracker
from measurement import Calibrator, MeasurementStore
from visualization import Visualizer

VIDEO = r"C:\Users\Süleyman\Desktop\dcb_test.mp4"
_c = _glob.glob(os.path.join(os.environ.get("USERPROFILE","C:\\Users\\*"), "Desktop", "dcb_test.mp4"))
if _c: VIDEO = _c[0]

ROI_RECT = ROI(x=100, y=80, w=470, h=260)
PX_PER_MM = 4.4
OUT = r"C:\dcb_frames"
os.makedirs(OUT, exist_ok=True)

SAVE_FRAMES = {50, 300, 500, 700, 900, 1050, 1080, 1300, 1550}

def main():
    reader = VideoReader(VIDEO)
    det = EdgeDetector(
        min_gap_pixels=3, min_run_pixels=12, min_threshold=110,
        min_arm_brightness=145.0, min_lower_brightness=130.0,
        max_gap_search=65, max_gap_pixels=90, max_lower_y_frac=0.82,
        mode="threshold",
    )
    trk = CrackTracker(min_gap_pixels=3, smooth_window=7, crack_direction="right")
    cal = Calibrator(pixels_per_mm=PX_PER_MM)
    store = MeasurementStore(fps=reader.fps, calibrator=cal)
    vis = Visualizer()

    reader.reset()
    for i in range(reader.frame_count):
        ret, frame = reader.read_frame()
        if not ret:
            break
        crop = frame[ROI_RECT.slice_yx]
        upper_l, lower_l = det.detect(crop)
        state = trk.update(i, upper_l, lower_l)
        tip_g = ROI_RECT.to_global(state.tip_x, state.tip_y)
        init_g = ROI_RECT.to_global(*trk.initial_tip) if trk.initial_tip else None
        upper_g = ROI_RECT.pts_to_global(upper_l)
        lower_g = ROI_RECT.pts_to_global(lower_l)
        m = store.record(i, state.propagation_px, tip_g, state.gap_at_tip)
        if i in SAVE_FRAMES:
            out_frame = vis.draw_overlay(
                frame, upper_g, lower_g, tip_g, init_g, m, reader.fps,
                (ROI_RECT.x, ROI_RECT.y, ROI_RECT.w, ROI_RECT.h),
            )
            cv2.imwrite(os.path.join(OUT, f"annot_{i:04d}.png"), out_frame)
            print(f"Frame {i:4d}: tip=({tip_g[0]},{tip_g[1]})  "
                  f"len={m.crack_length_mm:.2f}mm  gap={state.gap_at_tip}px")

    print(f"\nSaved to {OUT}")

if __name__ == "__main__":
    main()
