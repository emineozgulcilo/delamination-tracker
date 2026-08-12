#!/usr/bin/env python3
"""
diag_binary2.py - Check detection for frames 700 and 1200 with new code.
Prints per-column stats at several x positions and saves binary images.
"""
import sys, os, cv2, numpy as np, glob as _glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from video_reader import VideoReader
from roi import ROI
from edge_detection import EdgeDetector

VIDEO = r"C:\Users\Süleyman\Desktop\dcb_test.mp4"
_c = _glob.glob(os.path.join(os.environ.get("USERPROFILE","C:\\Users\\*"), "Desktop", "dcb_test.mp4"))
if _c: VIDEO = _c[0]

ROI_RECT = ROI(x=100, y=215, w=470, h=130)
OUT = r"C:\dcb_frames"
os.makedirs(OUT, exist_ok=True)

FRAMES_TO_SAVE = [700, 1200]

def analyse_frame(det, preprocessed, frame_idx):
    h, w = preprocessed.shape
    otsu_val, _ = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(int(otsu_val), det.min_threshold)
    _, binary = cv2.threshold(preprocessed, thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    print(f"\n=== Frame {frame_idx}  Otsu={otsu_val:.0f}  thresh={thresh} ===")

    # Sample columns at various x positions
    sample_xs = [100, 140, 180, 220, 240, 260, 300, 350, 400, 460]
    for x in sample_xs:
        if x >= w:
            continue
        col = binary[:, x]
        runs = det._white_runs(col, h)
        valid = [(s, e) for s, e in runs if (e - s + 1) >= det.min_run]
        col_g = preprocessed[:, x].astype(float)

        # Find bright runs
        bright = [(s, e) for s, e in valid
                  if float(np.mean(col_g[s:e+1])) >= det.min_arm_brightness]

        run_info = " ".join(f"[{s}-{e}:{np.mean(col_g[s:e+1]):.0f}]" for s,e in valid)
        bright_info = f" bright={len(bright)}"
        if len(bright) >= 2:
            gap = bright[1][0] - bright[0][1]
            bright_info += f" gap={gap}"
        print(f"  x={x:3d}: {len(valid)} runs {run_info}{bright_info}")

    return binary

def main():
    reader = VideoReader(VIDEO)
    det = EdgeDetector(
        min_gap_pixels=5,
        min_run_pixels=12,
        min_threshold=110,
        min_arm_brightness=145.0,
        mode="threshold",
    )

    target_set = set(FRAMES_TO_SAVE)
    reader.reset()
    for i in range(reader.frame_count):
        ret, frame = reader.read_frame()
        if not ret:
            break
        if i not in target_set:
            continue

        crop = frame[ROI_RECT.slice_yx]
        preprocessed = det.preprocess(crop)
        binary = analyse_frame(det, preprocessed, i)

        cv2.imwrite(os.path.join(OUT, f"bin2_{i:04d}.png"), binary)
        cv2.imwrite(os.path.join(OUT, f"roi2_{i:04d}.png"), crop)
        cv2.imwrite(os.path.join(OUT, f"pre2_{i:04d}.png"), preprocessed)

        target_set.discard(i)
        if not target_set:
            break

    print("\nSaved to", OUT)

if __name__ == "__main__":
    main()
