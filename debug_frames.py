"""
debug_frames.py  —  kare 897-903 arası ROI görüntüsü + current detection overlay kaydet
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\Süleyman\Desktop\delamınasyon")
from tracker.edge_detection import EdgeDetector

VIDEO = r"C:\Users\Süleyman\Desktop\dcb_test.mp4"
ROI_X, ROI_Y, ROI_W, ROI_H = 100, 80, 470, 260
OUTDIR = r"C:\Users\Süleyman\Desktop\delamınasyon\debug_frames"

import os; os.makedirs(OUTDIR, exist_ok=True)

detector = EdgeDetector(
    min_gap_pixels=3, clahe_clip=1.5,
    min_arm_brightness=145.0, min_lower_brightness=130.0,
    max_gap_search=65, max_gap_pixels=90,
    max_lower_y_frac=0.82, bright_crack=False, mode="threshold",
)

cap = cv2.VideoCapture(VIDEO)
frames_to_save = list(range(897, 906))

for fi in frames_to_save:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue

    roi_bgr = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    upper, lower = detector.detect(roi_gray)
    sp = detector._subpixel_tip_x

    # Overlay on BGR ROI
    vis = roi_bgr.copy()

    # Draw detected boundaries
    if upper is not None:
        for p in upper:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < ROI_W and 0 <= y < ROI_H:
                cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)
    if lower is not None:
        for p in lower:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < ROI_W and 0 <= y < ROI_H:
                cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)

    # Draw tip marker
    if upper is not None:
        tip_x = int(upper[-1, 0])
        tip_y = int(upper[-1, 1])
        cv2.drawMarker(vis, (tip_x, tip_y), (0, 0, 255),
                       cv2.MARKER_CROSS, 15, 2)

    # Mark 30 columns ahead in red vertical line
    if upper is not None:
        ahead_x = int(upper[-1, 0]) + 1
        if 0 <= ahead_x < ROI_W:
            cv2.line(vis, (ahead_x, 0), (ahead_x, ROI_H-1), (255, 0, 0), 1)

    cv2.putText(vis, f"F{fi}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    if upper is not None:
        cv2.putText(vis, f"tip={int(upper[-1,0])}", (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imwrite(f"{OUTDIR}/frame_{fi:04d}.png", vis)

    # Also save zoomed-in tip region (x=tip-30 to tip+60)
    if upper is not None:
        tx = int(upper[-1, 0])
        x0 = max(0, tx - 30)
        x1 = min(ROI_W, tx + 60)
        zoom = vis[:, x0:x1]
        zoom_big = cv2.resize(zoom, (zoom.shape[1]*3, zoom.shape[0]*3),
                               interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{OUTDIR}/zoom_{fi:04d}.png", zoom_big)

cap.release()
print(f"Frames saved to {OUTDIR}")
