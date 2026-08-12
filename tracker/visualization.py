"""
visualization.py
----------------
Wance-style DCB crack tracking overlay:
  - Blue square markers on upper/lower boundaries (every N columns)
  - Black vertical tick marks connecting boundaries at each column
  - Yellow horizontal centerline from initial tip to current tip
  - Magenta "+" crosshair at initial tip position (a0 reference)
  - "Cr0 c1| XX.XXXX| mm" measurement text
"""

import cv2
import numpy as np
from typing import Optional
try:
    from measurement import Measurement
except ModuleNotFoundError:
    from tracker.measurement import Measurement

# BGR color constants
_BLUE_MK   = (220,  60,  20)   # blue marker squares
_TICK      = ( 25,  25,  25)   # dark tick lines
_GREEN_ARR = (  0, 210,  80)   # lime-green centerline arrow (matches reference)
_PURPLE    = (210,  20, 210)   # magenta crosshair
_WHITE     = (255, 255, 255)
_BLACK     = (  0,   0,   0)
_CYAN      = (255, 255,   0)   # ROI rect


class Visualizer:
    """Wance-style DCB crack tracking visualizer."""

    def __init__(self, marker_step: int = 6, marker_half: int = 3):
        self.font        = cv2.FONT_HERSHEY_SIMPLEX
        self.marker_step = marker_step   # columns between blue square markers
        self.marker_half = marker_half   # half-size (px) of each square

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def draw_overlay(
        self,
        frame: np.ndarray,
        upper_pts_global: Optional[np.ndarray],
        lower_pts_global: Optional[np.ndarray],
        tip_global: tuple[int, int],
        initial_tip_global: Optional[tuple[int, int]],
        measurement: Optional[Measurement],
        fps: float,
        roi_rect: Optional[tuple[int, int, int, int]] = None,
    ) -> np.ndarray:

        out = (cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
               if len(frame.shape) == 2 else frame.copy())

        # ── Column lookup dicts ────────────────────────────────────────
        u_dict: dict[int, int] = {}
        l_dict: dict[int, int] = {}
        if upper_pts_global is not None and len(upper_pts_global) > 0:
            u_dict = {int(p[0]): int(p[1]) for p in upper_pts_global}
        if lower_pts_global is not None and len(lower_pts_global) > 0:
            l_dict = {int(p[0]): int(p[1]) for p in lower_pts_global}

        # ── Tick marks + blue square markers ──────────────────────────
        if u_dict and l_dict:
            common_xs = sorted(set(u_dict) & set(l_dict))
            s = self.marker_half
            # marker_interval is in absolute x-pixel coordinates so that
            # the same grid columns always get a square regardless of how
            # many columns were detected this frame.
            marker_interval = self.marker_step * 2   # every 12 px by default
            for x in common_xs:
                uy = u_dict[x]
                ly = l_dict[x]
                # Vertical tick every column
                cv2.line(out, (x, uy), (x, ly), _TICK, 1)
                # Square markers at fixed world-x positions
                if x % marker_interval == 0:
                    cv2.rectangle(out, (x - s, uy - s), (x + s, uy + s),
                                  _BLUE_MK, -1)
                    cv2.rectangle(out, (x - s, ly - s), (x + s, ly + s),
                                  _BLUE_MK, -1)

        # ── Yellow horizontal centerline with arrow ────────────────────
        valid_tip = (tip_global[0] != 0 or tip_global[1] != 0)
        has_init  = (initial_tip_global is not None
                     and (initial_tip_global[0] != 0 or initial_tip_global[1] != 0))

        if valid_tip and has_init:
            ix, iy = initial_tip_global
            tx, _ = tip_global
            tx = max(tx, ix)   # never draw a backward arrow
            length  = max(1, abs(tx - ix))
            tip_len = max(0.02, min(0.10, 18.0 / length))
            cv2.arrowedLine(out, (ix, iy), (tx, iy),
                            _GREEN_ARR, 2, cv2.LINE_AA, tipLength=tip_len)

        # ── Magenta "+" crosshair at initial tip (a0 marker) ──────────
        if has_init:
            ix, iy = initial_tip_global
            arm   = 14
            box_h = 10
            # Cross arms
            cv2.line(out, (ix - arm, iy), (ix + arm, iy), _PURPLE, 2)
            cv2.line(out, (ix, iy - arm), (ix, iy + arm), _PURPLE, 2)
            # Bounding box (Wance-style blue rectangle)
            cv2.rectangle(out,
                          (ix - arm - 4, iy - box_h),
                          (ix + arm + 4, iy + box_h),
                          _BLUE_MK, 1)
            # Measurement text just below the crosshair box
            if measurement is not None:
                txt = f"Cr0 c1|  {measurement.crack_length_mm:.4f}| mm"
                ty = iy + box_h + 18
                cv2.putText(out, txt, (ix - arm - 4 + 1, ty + 1),
                            self.font, 0.55, _BLACK, 3, cv2.LINE_AA)
                cv2.putText(out, txt, (ix - arm - 4, ty),
                            self.font, 0.55, _WHITE, 1, cv2.LINE_AA)

        # ── ROI outline (subtle) ───────────────────────────────────────
        if roi_rect is not None:
            x, y, w, h = roi_rect
            cv2.rectangle(out, (x, y), (x + w, y + h), _CYAN, 1)

        return out
