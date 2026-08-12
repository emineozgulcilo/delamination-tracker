#!/usr/bin/env python3
"""
diag_binary.py - Save binary threshold image and arm-brightness data for a
few frames so we can see why detections change.
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

FRAMES_TO_SAVE = [0, 100, 300]

def analyse_frame(det, preprocessed, frame_idx):
    """Print per-column info for columns near x=460-470 (false tip area)."""
    import cv2
    h, w = preprocessed.shape
    otsu_val, _ = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(int(otsu_val), det.min_threshold)
    _, binary = cv2.threshold(preprocessed, thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    print(f"\n--- Frame {frame_idx}  Otsu={otsu_val:.0f}  thresh={thresh} ---")
    # Check columns 90-115 (local, around the real crack tip area)
    # and 455-469 (local, the false tip area)
    for zone, xs in [("TIP ZONE (local 90-115)", range(90, 116)),
                     ("FALSE ZONE (local 455-469)", range(455, 470))]:
        hits = 0
        for x in xs:
            col = binary[:, x]
            runs = det._white_runs(col, h)
            valid = [(s, e) for s, e in runs if (e - s + 1) >= det.min_run]
            if len(valid) < 2:
                continue
            top_edge = valid[0][1]
            bot_edge = valid[-1][0]
            gap = bot_edge - top_edge
            if gap < det.min_gap:
                continue
            col_g = preprocessed[:, x].astype(float)
            arm1 = float(np.mean(col_g[valid[0][0]:valid[0][1]+1]))
            arm2 = float(np.mean(col_g[valid[-1][0]:valid[-1][1]+1]))
            ratio = min(arm1, arm2) / max(arm1, arm2) if max(arm1, arm2) > 0 else 0
            sym_pass = ratio >= det.min_arm_symmetry
            hits += 1
            if hits <= 3:  # print first few
                print(f"  col x={x:3d}  gap={gap:3d}px  "
                      f"arm1={arm1:.0f}  arm2={arm2:.0f}  "
                      f"ratio={ratio:.2f}  sym={'PASS' if sym_pass else 'FAIL'}")
        if hits > 3:
            print(f"  ... {hits} total detections in {zone}")
        elif hits == 0:
            print(f"  No 2-run columns in {zone}")
    return binary

def main():
    reader = VideoReader(VIDEO)
    det = EdgeDetector(
        min_gap_pixels=5,
        min_run_pixels=12,
        min_threshold=110,
        min_arm_symmetry=0.70,
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

        cv2.imwrite(os.path.join(OUT, f"bin_{i:04d}.png"), binary)
        cv2.imwrite(os.path.join(OUT, f"roi_{i:04d}.png"), crop)
        cv2.imwrite(os.path.join(OUT, f"pre_{i:04d}.png"), preprocessed)

        if target_set == {i}: break  # optimisation doesn't work for sets like this
        target_set.discard(i)
        if not target_set:
            break

    print("\nImages saved to", OUT)

if __name__ == "__main__":
    main()
