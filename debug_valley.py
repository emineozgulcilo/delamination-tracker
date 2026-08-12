"""
debug_valley.py  —  frame 870-880 arasi kolom profilleri incele
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\Süleyman\Desktop\delamınasyon")

from tracker.edge_detection import EdgeDetector, _smooth1d

VIDEO = r"C:\Users\Süleyman\Desktop\dcb_test.mp4"
ROI_X, ROI_Y, ROI_W, ROI_H = 100, 80, 470, 260

detector = EdgeDetector(
    min_gap_pixels=3,
    clahe_clip=1.5,
    min_arm_brightness=145.0,
    min_lower_brightness=130.0,
    max_gap_search=65,
    max_gap_pixels=90,
    max_lower_y_frac=0.82,
    bright_crack=False,
    mode="threshold",
)

cap = cv2.VideoCapture(VIDEO)
frames_to_check = [860, 865, 870, 875, 880]

for fi in frames_to_check:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue

    roi_crop = cv2.cvtColor(frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W], cv2.COLOR_BGR2GRAY)
    upper, lower = detector.detect(roi_crop)
    sp = detector._subpixel_tip_x

    # preprocess ile ayni goruntu (roi_crop zaten gray)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    prep = clahe.apply(roi_crop)

    tip_x = int(upper[-1, 0]) if upper is not None else -1
    print(f"\n=== Frame {fi}  tip_x(local)={tip_x}  sp={sp} ===")
    if upper is None:
        print("  No detection")
        continue

    # 5 kolon ilerisi icin profil goster
    print(f"  Scanning 5 cols after tip:")
    for dx in range(1, 6):
        x = tip_x + dx
        if x >= ROI_W:
            break
        col = prep[:, x].astype(float)
        col_s = _smooth1d(col, sigma=1.5)

        # beklenen upper/lower y (polinom tahmini degil, son bilinen)
        uy = int(upper[-1, 1])
        ly = int(lower[-1, 1])

        WIN = 14
        y0 = max(0, uy - WIN)
        y1 = min(ROI_H - 1, ly + WIN)
        profile = col_s[y0:y1+1]
        n = len(profile)

        u_end = min(n-1, (uy - y0) + WIN)
        l_start = max(0, (ly - y0) - WIN)

        if u_end < 1 or l_start >= n-1 or l_start <= u_end:
            print(f"    x={x}: invalid search range u_end={u_end} l_start={l_start}")
            continue

        u_idx = int(np.argmax(profile[:u_end+1]))
        u_val = float(profile[u_idx])
        l_idx = int(np.argmax(profile[l_start:])) + l_start
        l_val = float(profile[l_idx])

        if l_idx <= u_idx:
            print(f"    x={x}: peaks in wrong order (u={u_idx} l={l_idx})")
            continue

        arm_mean = (u_val + l_val) / 2.0
        between = profile[u_idx:l_idx+1]
        v_val = float(between[np.argmin(between)])
        valley_frac = (arm_mean - v_val) / arm_mean if arm_mean > 0 else 0
        abs_dip = arm_mean - v_val

        print(f"    x={x}: u_val={u_val:.1f} l_val={l_val:.1f} v_val={v_val:.1f} "
              f"arm_mean={arm_mean:.1f} valley_frac={valley_frac:.3f} abs_dip={abs_dip:.1f} "
              f"min_thresh={145*0.75:.1f}")

cap.release()
