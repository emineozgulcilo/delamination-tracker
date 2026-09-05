"""
gap_tracker.py — DCB çatlak ucu takibi.
Kırmızı çizgi: çatlak ucu (en sağ gap sütunu, monoton).
Crack: tip - initial_x (warmup ile sabitlenir).
"""
import cv2
import numpy as np
import csv
import time
import statistics
from pathlib import Path

VIDEO_IN  = "C:/Users/Süleyman/Desktop/dcb_test_clean.mp4"
VIDEO_OUT = "C:/Users/Süleyman/Desktop/dcb_delamination/gap_tracked.mp4"
CSV_OUT   = "output_gap/measurements_gap.csv"

ROI      = (100, 80, 470, 260)
PPM      = 4.4
THRESH   = 90
GAP_MIN  = 4
MIN_WHITE = 8
X_MIN    = 30
X_MAX    = 420
WARMUP   = 30

Path("output_gap").mkdir(exist_ok=True)

_RED   = (0, 0, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_CYAN  = (255, 255, 0)
FONT   = cv2.FONT_HERSHEY_SIMPLEX


def gap_profile(roi_gray):
    h, w = roi_gray.shape
    _, bw = cv2.threshold(roi_gray, THRESH, 255, cv2.THRESH_BINARY)
    gaps = np.zeros(w, dtype=int)
    for col in range(w):
        col_vals = bw[:, col]
        runs, in_run, start = [], False, 0
        for i, v in enumerate(col_vals):
            if v == 255:
                if not in_run: in_run, start = True, i
            else:
                if in_run:
                    in_run = False
                    if i - start >= MIN_WHITE:
                        runs.append((start, i - 1))
        if in_run and (h - start) >= MIN_WHITE:
            runs.append((start, h - 1))
        if len(runs) >= 2:
            g = runs[-1][0] - runs[0][1] - 1
            gaps[col] = max(0, g)
    return gaps


def find_tip_raw(gaps):
    for col in range(min(X_MAX, len(gaps) - 1), X_MIN - 1, -1):
        if gaps[col] >= GAP_MIN:
            return col
    return -1


cap    = cv2.VideoCapture(VIDEO_IN)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

max_seen  = 0
initial_x = None
warmup_xs = []

rx, ry, rw, rh = ROI
records = []
t0 = time.time()

print(f"[Basliyor] {total} kare  {fps:.1f}fps")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi_gray = cv2.cvtColor(frame[ry:ry+rh, rx:rx+rw], cv2.COLOR_BGR2GRAY)
    gaps     = gap_profile(roi_gray)
    tip_raw  = find_tip_raw(gaps)

    if tip_raw > 0:
        max_seen = max(max_seen, tip_raw)

    if frame_idx < WARMUP and tip_raw > 0:
        warmup_xs.append(tip_raw)
    if frame_idx == WARMUP and warmup_xs:
        initial_x = int(statistics.median(warmup_xs))
        print(f"[WARMUP] initial_x={initial_x}")

    tip_roi    = max_seen if max_seen > 0 else 0
    tip_global = rx + tip_roi if tip_roi > 0 else 0

    propagation_px = max(0, tip_roi - initial_x) if initial_x else 0
    crack_mm       = propagation_px / PPM

    out = frame.copy()

    if tip_global > 0:
        cv2.line(out, (tip_global, 0), (tip_global, height - 1), _RED, 2, cv2.LINE_AA)

    tx = max(4, tip_global - 130)
    cv2.putText(out, f"Crack: {crack_mm:.2f} mm", (tx+1, 29), FONT, 0.60, _BLACK, 3, cv2.LINE_AA)
    cv2.putText(out, f"Crack: {crack_mm:.2f} mm", (tx,   28), FONT, 0.60, _WHITE, 1, cv2.LINE_AA)

    cv2.rectangle(out, (rx, ry), (rx+rw, ry+rh), _CYAN, 1)

    writer.write(out)
    records.append((frame_idx, frame_idx/fps, crack_mm, tip_global))

    if frame_idx % 100 == 0:
        pct = frame_idx / total * 100
        print(f"  {frame_idx:>5}/{total}  ({pct:4.1f}%)  tip={tip_roi}  crack={crack_mm:.2f}mm")

    frame_idx += 1

cap.release()
writer.release()

with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Frame", "Time(s)", "CrackLength(mm)", "CrackTipX"])
    w.writerows(records)

print(f"\n[Bitti] {frame_idx} kare  {time.time()-t0:.1f}s")
print(f"[Video] {VIDEO_OUT}")
