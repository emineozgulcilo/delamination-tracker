"""
gap_tracker.py — DCB çatlak ucu takibi.
Kırmızı çizgi: çatlak ucu (en sağ gap sütunu, monoton).
Yeşil nokta: alt kolun üst yüzeyi, görünür bölgede.
Crack ve Arc: mm cinsinden.
"""
import cv2
import numpy as np
import csv
import time
import statistics
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

VIDEO_IN  = "C:/Users/Süleyman/Desktop/dcb_test_clean.mp4"
VIDEO_OUT = "C:/Users/Süleyman/Desktop/dcb_delamination/gap_tracked.mp4"
CSV_OUT   = "output_gap/measurements_gap.csv"

ROI       = (100, 80, 470, 260)
PPM       = 4.4
THRESH    = 90
THRESH_SURF = 60
GAP_MIN   = 4
MIN_WHITE = 8
X_MIN     = 30
X_MAX     = 420
WARMUP    = 30
SMOOTH_WIN = 15
MAX_GAP_FILL = 8

Path("output_gap").mkdir(exist_ok=True)

_RED   = (0, 0, 220)
_GREEN = (0, 220, 0)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_CYAN  = (255, 255, 0)
FONT   = cv2.FONT_HERSHEY_SIMPLEX


def _col_runs(col_vals, min_white):
    runs, in_run, start = [], False, 0
    n = len(col_vals)
    for i, v in enumerate(col_vals):
        if v == 255:
            if not in_run:
                in_run, start = True, i
        else:
            if in_run:
                in_run = False
                if i - start >= min_white:
                    runs.append((start, i - 1))
    if in_run and (n - start) >= min_white:
        runs.append((start, n - 1))
    return runs


def gap_profile(roi_gray):
    h, w = roi_gray.shape
    _, bw  = cv2.threshold(roi_gray, THRESH,       255, cv2.THRESH_BINARY)
    _, bw2 = cv2.threshold(roi_gray, THRESH_SURF,  255, cv2.THRESH_BINARY)
    gaps      = np.zeros(w, dtype=int)
    lower_top = np.full(w, -1, dtype=int)

    for col in range(w):
        runs = _col_runs(bw[:, col], MIN_WHITE)
        if len(runs) >= 2:
            g = runs[-1][0] - runs[0][1] - 1
            gaps[col] = max(0, g)
            g2 = runs[1][0] - runs[0][1] - 1
            if g2 >= 1:
                lower_top[col] = runs[1][0]

        if lower_top[col] < 0:
            r2 = _col_runs(bw2[:, col], MIN_WHITE)
            if len(r2) >= 2:
                g2 = r2[1][0] - r2[0][1] - 1
                if g2 >= 1:
                    lower_top[col] = r2[1][0]

    return gaps, lower_top


def get_arc_pts(lower_top, x_left, x_right):
    cols = np.arange(x_left, x_right + 1)
    raw  = lower_top[x_left:x_right + 1].astype(float)
    raw[raw < 0] = np.nan

    valid_idx = np.where(~np.isnan(raw))[0]
    if len(valid_idx) < 2:
        return []

    filled = raw.copy()
    for i in range(len(valid_idx) - 1):
        a, b = valid_idx[i], valid_idx[i + 1]
        if 0 < (b - a - 1) <= MAX_GAP_FILL:
            for j in range(a + 1, b):
                t = (j - a) / (b - a)
                filled[j] = raw[a] * (1 - t) + raw[b] * t

    valid_f = ~np.isnan(filled)
    if valid_f.sum() < 2:
        return []

    fx = cols[valid_f]
    fy = filled[valid_f]

    pad      = SMOOTH_WIN
    padded   = np.pad(fy, pad, mode='edge')
    wins     = sliding_window_view(padded, 2 * pad + 1)
    smoothed = np.median(wins, axis=1)

    return list(zip(fx, smoothed))


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

    roi_gray        = cv2.cvtColor(frame[ry:ry+rh, rx:rx+rw], cv2.COLOR_BGR2GRAY)
    gaps, lower_top = gap_profile(roi_gray)
    tip_raw         = find_tip_raw(gaps)

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

    # Yeşil arc: initial_x'ten tip'e kadar görünür bölge
    arc_mm   = 0.0
    surf_pts = []

    if initial_x and tip_roi > initial_x + 2:
        pts = get_arc_pts(lower_top, initial_x, tip_roi)
        if len(pts) >= 2:
            arc_px = sum(
                np.sqrt((pts[i][0] - pts[i-1][0])**2 + (pts[i][1] - pts[i-1][1])**2)
                for i in range(1, len(pts))
            )
            arc_mm = arc_px / PPM
            for col, y_roi in pts[::2]:
                gx = rx + int(col)
                gy = ry + int(round(y_roi))
                if 0 <= gy < height:
                    surf_pts.append((gx, gy))

    out = frame.copy()

    for pt in surf_pts:
        cv2.circle(out, pt, 2, _GREEN, -1, cv2.LINE_AA)

    if tip_global > 0:
        cv2.line(out, (tip_global, 0), (tip_global, height - 1), _RED, 2, cv2.LINE_AA)

    tx = max(4, tip_global - 130)
    cv2.putText(out, f"Crack: {crack_mm:.2f} mm", (tx+1, 29), FONT, 0.60, _BLACK, 3, cv2.LINE_AA)
    cv2.putText(out, f"Crack: {crack_mm:.2f} mm", (tx,   28), FONT, 0.60, _WHITE, 1, cv2.LINE_AA)
    cv2.putText(out, f"Arc:   {arc_mm:.2f} mm",   (tx+1, 52), FONT, 0.60, _BLACK, 3, cv2.LINE_AA)
    cv2.putText(out, f"Arc:   {arc_mm:.2f} mm",   (tx,   51), FONT, 0.60, _GREEN, 1, cv2.LINE_AA)

    cv2.rectangle(out, (rx, ry), (rx+rw, ry+rh), _CYAN, 1)

    writer.write(out)
    records.append((frame_idx, frame_idx/fps, crack_mm, tip_global, arc_mm))

    if frame_idx % 100 == 0:
        pct = frame_idx / total * 100
        print(f"  {frame_idx:>5}/{total}  ({pct:4.1f}%)  tip={tip_roi}  crack={crack_mm:.2f}mm  arc={arc_mm:.2f}mm")

    frame_idx += 1

cap.release()
writer.release()

with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Frame", "Time(s)", "CrackLength(mm)", "CrackTipX", "ArcLength(mm)"])
    w.writerows(records)

print(f"\n[Bitti] {frame_idx} kare  {time.time()-t0:.1f}s")
print(f"[Video] {VIDEO_OUT}")
