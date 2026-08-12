"""
debug_frames2.py  —  hazır annotated video'dan frame çıkar
"""
import cv2, os

VIDEO = r"C:\dcb_sonuclar_valley\annotated_video.mp4"
OUTDIR = r"C:\Users\Süleyman\Desktop\delamınasyon\debug_frames"
os.makedirs(OUTDIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
# Jump öncesi ve sonrası: 895-908, ayrıca ilerleme fazı 60-90
frames_to_save = list(range(60, 73)) + list(range(895, 909))

for fi in frames_to_save:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"{OUTDIR}/ann_{fi:04d}.png", frame)
        # ROI crop zoom
        roi = frame[80:340, 100:570]
        big = cv2.resize(roi, (roi.shape[1]*2, roi.shape[0]*2), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f"{OUTDIR}/roi_{fi:04d}.png", big)

cap.release()
print(f"Saved to {OUTDIR}")
