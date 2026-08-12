#!/usr/bin/env python3
"""
diag_v15.py  –  Frame 900 için boundary tespitini görsel olarak göster.
Çalıştır:  python tracker/diag_v15.py
"""
import sys, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from video_reader import VideoReader
from roi import ROI
from edge_detection import EdgeDetector

VIDEO   = r"C:\Users\Süleyman\Desktop\dcb_test.mp4"
FRAME   = 900          # kontrol edilecek frame
ROI_R   = ROI(x=100, y=80, w=470, h=260)
OUT     = r"C:\dcb_diag_v15"
import os; os.makedirs(OUT, exist_ok=True)

def main():
    reader = VideoReader(VIDEO)
    det = EdgeDetector(
        min_gap_pixels=3,
        clahe_clip=1.5,
        min_arm_brightness=145.0,
        min_lower_brightness=130.0,
        max_gap_search=65,
        mode="threshold",
    )

    # İlk FRAME kareye git
    reader.reset()
    for i in range(FRAME + 1):
        ret, frame = reader.read_frame()
        if not ret:
            break

    crop        = frame[ROI_R.slice_yx]
    preprocessed = det.preprocess(crop)
    upper, lower = det.detect(crop)

    # ── Görsel 1: CLAHE önişlem ────────────────────────────────────
    cv2.imwrite(f"{OUT}/pre_{FRAME:04d}.png", preprocessed)

    # ── Görsel 2: Tespitli boundary ───────────────────────────────
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if upper is not None and lower is not None:
        u_dict = {int(x): int(y) for x, y in upper}
        l_dict = {int(x): int(y) for x, y in lower}
        for xi in sorted(set(u_dict) & set(l_dict)):
            uy, ly = u_dict[xi], l_dict[xi]
            cv2.line(vis, (xi, uy), (xi, ly), (200, 50, 50), 1)
            if xi % 20 == 0:
                cv2.circle(vis, (xi, uy), 3, (0, 0, 255), -1)  # kırmızı = üst
                cv2.circle(vis, (xi, ly), 3, (0, 255, 0), -1)  # yeşil = alt
    cv2.imwrite(f"{OUT}/det_{FRAME:04d}.png", vis)

    # ── Görsel 3: Sütun x=200 detaylı brightness profili ──────────
    col_x = 200
    col_g = preprocessed[:, col_x].astype(float)
    h = preprocessed.shape[0]

    # binary hesapla
    otsu_val, _ = cv2.threshold(preprocessed, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(int(otsu_val), 110)
    _, binary = cv2.threshold(preprocessed, thresh, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{OUT}/bin_{FRAME:04d}.png", binary)

    print(f"Frame {FRAME} – Otsu={otsu_val:.0f}  thresh={thresh}")
    print(f"Sütun x={col_x} brightness (y=80..180):")
    for y in range(80, min(181, h)):
        bar = "#" * int(col_g[y] / 5)
        marker = ""
        if upper is not None and col_x in {int(p[0]) for p in upper}:
            ud = {int(p[0]): int(p[1]) for p in upper}
            ld = {int(p[0]): int(p[1]) for p in lower}
            if y == ud.get(col_x):   marker = " <<< ÜST ARM SINIRI"
            if y == ld.get(col_x):   marker = " <<< ALT ARM SINIRI"
        print(f"  y={y:3d}  {col_g[y]:5.0f}  {bar}{marker}")

    if upper is not None:
        print(f"\nToplam üst nokta: {len(upper)}, alt nokta: {len(lower)}")
        print(f"x aralığı: {int(upper[0,0])}–{int(upper[-1,0])}")
    print(f"\nGörseller kaydedildi: {OUT}")

if __name__ == "__main__":
    main()
