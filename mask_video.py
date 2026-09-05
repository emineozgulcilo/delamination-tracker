"""
mask_video.py — Videodaki metin overlay'lerini siyahla kapatır, temiz kopyayı kaydeder.
"""
import cv2
import numpy as np

VIDEO_IN  = "C:/Users/S\u00fcleyman/Desktop/dcb_test.mp4"
VIDEO_OUT = "C:/Users/S\u00fcleyman/Desktop/dcb_delamination/dcb_test_clean.mp4"

# Kapatılacak bölgeler: (y0, y1, x0, x1) — global frame koordinatları
MASKS = [
    (1,  17,   3, 340),   # üst sol: "WMC frame 2990 Disp..."
    (31, 105, 204, 630),  # sağ üst metin bloğu (DCB test bilgileri)
    (272, 335, 346, 623), # sağ alt: WMC Knowledge Centre logo
]

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Video açılamadı: {VIDEO_IN}")

fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

print(f"[Kaynak]  {VIDEO_IN}")
print(f"[Hedef]   {VIDEO_OUT}")
print(f"[Video]   {width}x{height}  {fps:.2f}fps  {total} kare")
print(f"[Maskeler] {len(MASKS)} bölge kapatılıyor\n")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    for y0, y1, x0, x1 in MASKS:
        frame[y0:y1, x0:x1] = 0   # siyah ile kapat

    out.write(frame)
    frame_idx += 1

    if frame_idx % 100 == 0:
        pct = frame_idx / total * 100
        print(f"  {frame_idx}/{total}  ({pct:.1f}%)")

cap.release()
out.release()
print(f"\n[Bitti] {frame_idx} kare yazıldı → {VIDEO_OUT}")
