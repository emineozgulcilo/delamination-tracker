"""
measurement.py
--------------
Calibration and measurement storage.

Calibration converts pixel distances to millimetres.  Two modes:
  • Direct: user provides *pixels_per_mm* on the command line.
  • Interactive: user clicks two points with a known physical distance.

All per-frame measurements are stored in a flat list for CSV export.
"""

from __future__ import annotations

import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Measurement:
    """Single-frame measurement record."""

    frame: int
    time_s: float
    crack_length_mm: float
    crack_length_px: float
    crack_tip_x: int   # global pixel coordinates
    crack_tip_y: int
    gap_at_tip_px: int


class Calibrator:
    """
    Handles pixels-per-mm calibration.

    Usage (interactive mode)
    ------------------------
    cal = Calibrator()
    cal.select_points(frame)        # opens window, user clicks 2 pts
    cal.set_known_distance(25.4)    # user enters real distance in mm
    print(cal.pixels_per_mm)
    """

    def __init__(self, pixels_per_mm: float = 1.0):
        self.pixels_per_mm: float = pixels_per_mm
        self._points: list[tuple[int, int]] = []
        self._display: Optional[np.ndarray] = None

    def select_points(self, frame: np.ndarray) -> Optional[float]:
        """
        Open an interactive window.  User clicks two points with a known
        distance; after pressing ENTER the function computes the pixel
        distance and prompts (via stdout) for the real distance in mm.

        Returns the computed pixels_per_mm, or None if the user cancels.
        """
        self._points.clear()
        self._display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) \
            if len(frame.shape) == 2 else frame.copy()

        title = "Calibration — click 2 points, press ENTER to confirm"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(title, self._on_click)

        print("\n[Calibration] Click two points on a feature with a known distance.")
        print("             Press ENTER to confirm, ESC to skip.")

        while True:
            cv2.imshow(title, self._display)
            key = cv2.waitKey(50) & 0xFF
            if key in (13, 10) and len(self._points) == 2:   # ENTER
                break
            if key == 27:  # ESC
                cv2.destroyWindow(title)
                return None

        cv2.destroyWindow(title)

        px_dist = self._pixel_distance()
        try:
            mm = float(input("Enter the known distance in mm: ").strip())
            if mm <= 0:
                raise ValueError
        except ValueError:
            print("[Calibration] Invalid value — calibration skipped.")
            return None

        self.pixels_per_mm = px_dist / mm
        print(f"[Calibration] {self.pixels_per_mm:.4f} px/mm")
        return self.pixels_per_mm

    def _on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self._points) < 2:
            self._points.append((x, y))
            cv2.circle(self._display, (x, y), 6, (0, 0, 255), -1)
            if len(self._points) == 2:
                cv2.line(self._display, self._points[0], self._points[1],
                         (0, 255, 255), 2)
            cv2.imshow("Calibration — click 2 points, press ENTER to confirm",
                       self._display)

    def _pixel_distance(self) -> float:
        if len(self._points) < 2:
            return 1.0
        dx = self._points[1][0] - self._points[0][0]
        dy = self._points[1][1] - self._points[0][1]
        return math.sqrt(dx * dx + dy * dy)

    def px_to_mm(self, pixels: float) -> float:
        return pixels / self.pixels_per_mm if self.pixels_per_mm else pixels


class MeasurementStore:
    """Accumulates per-frame Measurement records."""

    def __init__(self, fps: float, calibrator: Calibrator,
                 initial_crack_mm: float = 0.0):
        self.fps = fps
        self.calibrator = calibrator
        self.initial_crack_mm = initial_crack_mm   # a0 offset (pre-crack length)
        self._records: list[Measurement] = []

    def record(
        self,
        frame_index: int,
        crack_length_px: float,
        tip_global: tuple[int, int],
        gap_at_tip_px: int,
    ) -> Measurement:
        prop_mm   = self.calibrator.px_to_mm(crack_length_px)
        total_mm  = self.initial_crack_mm + prop_mm   # a = a0 + da
        m = Measurement(
            frame=frame_index,
            time_s=round(frame_index / self.fps, 5),
            crack_length_px=round(crack_length_px, 3),
            crack_length_mm=round(total_mm, 4),
            crack_tip_x=tip_global[0],
            crack_tip_y=tip_global[1],
            gap_at_tip_px=gap_at_tip_px,
        )
        self._records.append(m)
        return m

    @property
    def records(self) -> list[Measurement]:
        return self._records

    @property
    def latest(self) -> Optional[Measurement]:
        return self._records[-1] if self._records else None
