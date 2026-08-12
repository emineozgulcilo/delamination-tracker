"""
debug_jump.py  —  sıçrama öncesi karelerde kritik kolonların intensite profilini çiz
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, r"C:\Users\Süleyman\Desktop\delamınasyon")
from tracker.edge_detection import _smooth1d

VIDEO = r"C:\Users\Süleyman\Desktop\dcb_test.mp4"
ROI_X, ROI_Y, ROI_W, ROI_H = 100, 80, 470, 260

cap = cv2.VideoCapture(VIDEO)

# Sıçrama öncesi: frames 895-905
# X bölgesi: tip_x=222 olduğunda ilerideki kolonlara bak (x_local = 230, 250, 270, 290, 309)
check_xs = [230, 250, 260, 270, 280, 290, 295, 300, 305, 309]
check_frames = list(range(895, 908))

# Her frame için her x kolunu al, max brightness değerini kaydet
data = {}  # frame → {x: (u_peak, valley, l_peak)}
for fi in check_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    prep = clahe.apply(roi_gray)

    data[fi] = {}
    for x in check_xs:
        if x >= ROI_W:
            continue
        col_s = _smooth1d(prep[:, x].astype(float), sigma=1.5)
        # Ara nötr: ROI orta bandı (y=90-180 approx)
        band = col_s[80:200]
        n = len(band)
        mid = n // 2
        u_peak = float(np.max(band[:mid]))
        l_peak = float(np.max(band[mid:]))
        valley = float(np.min(band[mid//2 : mid + mid//2]))
        arm_mean = (u_peak + l_peak) / 2.0
        vf = (arm_mean - valley) / arm_mean if arm_mean > 0 else 0
        data[fi][x] = (u_peak, valley, l_peak, vf)

cap.release()

# Tablo yazdır
print(f"{'Frame':>6} | " + " | ".join(f"x={x:3d}" for x in check_xs))
print("-" * (8 + len(check_xs) * 12))
for fi in check_frames:
    row = []
    for x in check_xs:
        if x in data.get(fi, {}):
            u, v, l, vf = data[fi][x]
            row.append(f"{vf*100:5.1f}%")
        else:
            row.append("   -  ")
    print(f"  {fi:4d}  | " + " | ".join(row))

print("\nlegend: valley_frac = (arm_mean - valley) / arm_mean  (>5% = gap visible)")
print("min_arm_brightness threshold at 0.75 × 145 = 108.75")
print()
print("u_peak / valley / l_peak values for key frames:")
for fi in [897, 898, 899, 900, 901]:
    row = []
    for x in [270, 280, 290, 300, 309]:
        if x in data.get(fi, {}):
            u, v, l, vf = data[fi][x]
            row.append(f"x={x}:({u:.0f}/{v:.0f}/{l:.0f}={vf*100:.1f}%)")
    print(f"  Frame {fi}: " + "  ".join(row))
