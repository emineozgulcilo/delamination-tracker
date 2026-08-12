"""
debug_valley2.py  —  sıçrama öncesi karelerde valley metodunun ne bulduğunu göster
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\Süleyman\Desktop\delamınasyon")
from tracker.edge_detection import EdgeDetector, _smooth1d

VIDEO = r"C:\Users\Süleyman\Desktop\dcb_test.mp4"
ROI_X, ROI_Y, ROI_W, ROI_H = 100, 80, 470, 260

detector = EdgeDetector(
    min_gap_pixels=3, clahe_clip=1.5,
    min_arm_brightness=145.0, min_lower_brightness=130.0,
    max_gap_search=65, max_gap_pixels=90,
    max_lower_y_frac=0.82, bright_crack=False, mode="threshold",
)

cap = cv2.VideoCapture(VIDEO)

# Çeşitli sıçrama öncesi kareler: 890-905 (first jump), 1055-1075 (second big jump)
frames_to_check = list(range(888, 908)) + list(range(1058, 1078))

prev_len = 0.0
for fi in frames_to_check:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue

    roi_gray = cv2.cvtColor(frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W], cv2.COLOR_BGR2GRAY)
    upper, lower = detector.detect(roi_gray)
    sp = detector._subpixel_tip_x

    tip_x = int(upper[-1, 0]) if upper is not None else -1
    tip_y = int(upper[-1, 1]) if upper is not None else -1

    # approximate propagation in px
    mm = (tip_x - 168) / 4.4 if tip_x > 0 else 0  # 168 = approx initial tip x
    print(f"Frame {fi:5d}: tip_x={tip_x:4d}  a≈{mm:6.2f}mm  sp={sp!r}")

cap.release()
