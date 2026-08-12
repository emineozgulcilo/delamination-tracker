"""
roi.py
------
Region of Interest data structure and interactive selector.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class ROI:
    """Rectangular region of interest in pixel coordinates."""

    x: int  # left edge
    y: int  # top edge
    w: int  # width
    h: int  # height

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def slice_yx(self) -> tuple[slice, slice]:
        """NumPy slice for frame[roi.slice_yx]."""
        return slice(self.y, self.y2), slice(self.x, self.x2)

    def to_global(self, local_x: int | float, local_y: int | float) -> tuple[int, int]:
        """Convert ROI-local pixel coords to full-frame coords."""
        return (int(self.x + local_x), int(self.y + local_y))

    def to_local(self, global_x: int, global_y: int) -> tuple[int, int]:
        """Convert full-frame coords to ROI-local coords."""
        return (global_x - self.x, global_y - self.y)

    def pts_to_global(self, pts: np.ndarray) -> np.ndarray:
        """Shift an Nx2 (x, y) array from local to global coordinates."""
        if pts is None or len(pts) == 0:
            return pts
        shifted = pts.astype(float).copy()
        shifted[:, 0] += self.x
        shifted[:, 1] += self.y
        return shifted.astype(int)

    def draw_on(self, frame: np.ndarray, color=(0, 255, 255), thickness=2) -> np.ndarray:
        """Draw the ROI rectangle on a BGR frame."""
        out = frame.copy()
        cv2.rectangle(out, (self.x, self.y), (self.x2, self.y2), color, thickness)
        return out


class ROISelector:
    """
    Lets the user draw a rectangular ROI on the first video frame.
    Uses cv2.selectROI and scales the display so it fits on screen.
    """

    MAX_DISPLAY_W = 1400
    MAX_DISPLAY_H = 900

    def select(self, frame: np.ndarray) -> ROI:
        """
        Open an interactive window and return the selected ROI.
        Raises ValueError if the user cancels (presses C) or selects nothing.
        """
        display, scale = self._fit_for_display(frame)

        window = "Select ROI — drag to draw, press ENTER/SPACE to confirm, C to cancel"
        rect = cv2.selectROI(window, display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window)

        if rect == (0, 0, 0, 0):
            raise ValueError("No ROI selected.")

        x, y, w, h = (int(v / scale) for v in rect)

        # Clamp to frame bounds
        x = max(0, min(x, frame.shape[1] - 2))
        y = max(0, min(y, frame.shape[0] - 2))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        return ROI(x=x, y=y, w=w, h=h)

    def _fit_for_display(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = min(self.MAX_DISPLAY_W / w, self.MAX_DISPLAY_H / h, 1.0)
        if scale < 1.0:
            nw, nh = int(w * scale), int(h * scale)
            display = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            display = frame.copy()

        # Convert to BGR for the selector window
        if len(display.shape) == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        return display, scale
