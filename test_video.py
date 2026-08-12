import cv2, sys
sys.path.insert(0, r"C:\Users\Suleyman\Desktop\delaminasyon")
sys.path.insert(0, r"C:\Users\Süleyman\Desktop\delamınasyon")
from video_analyzer import parse_wmc_line, get_reader

VIDEO = r"C:\Users\Süleyman\Desktop\dcb_test.mp4"
cap   = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
print("Video: %d kare, %.1f fps" % (total, fps))

reader = get_reader()

test_frames = [0, 100, 300, 500, 700, 900, 1100, 1300, total-2]
print("\n--- OCR Test ---")
for fidx in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if not ret:
        continue
    h = frame.shape[0]
    strip = frame[:int(h*0.15), :]
    gray  = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    big   = cv2.resize(th, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(big, detail=0)
    text = " ".join(results)
    parsed = parse_wmc_line(text)
    print("Kare %5d: [%s]" % (fidx, text[:90]))
    print("           Parse -> %s" % str(parsed))

cap.release()
print("\nTest tamamlandi.")
