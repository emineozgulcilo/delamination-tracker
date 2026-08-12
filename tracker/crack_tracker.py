"""
crack_tracker.py
----------------
Tracks the crack tip and computes propagation length across frames.

Crack tip definition
~~~~~~~~~~~~~~~~~~~~
The crack tip is the extremal column (rightmost for right-propagating cracks,
leftmost for left-propagating) at which the vertical gap between the two
detected boundaries exceeds *min_gap_pixels*.

Temporal smoothing
~~~~~~~~~~~~~~~~~~
Raw tip positions are noisy due to detection artifacts, small vibrations, and
lighting changes.  A rolling-median filter over the last *smooth_window* frames
is applied to produce a stable, jitter-free tip position.

Propagation length
~~~~~~~~~~~~~~~~~~
Measured as the Euclidean distance (in pixels) from the *initial tip* —
the smoothed tip position at the first frame where it is confidently detected —
to the current smoothed tip.  The horizontal component is also stored separately
for cases where only in-plane propagation is relevant.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameState:
    """All tracked quantities for a single processed frame."""

    frame_index: int
    tip_x: int                             # ROI-local x of crack tip (integer)
    tip_y: int                             # ROI-local y of crack tip
    upper_boundary: Optional[np.ndarray]   # Nx2 local coords
    lower_boundary: Optional[np.ndarray]   # Nx2 local coords
    propagation_px: float                  # Euclidean distance from initial tip
    propagation_h_px: float                # Horizontal component only
    gap_at_tip: int                        # Gap width (px) at the crack tip col
    tip_x_subpixel: float = 0.0            # Sub-pixel refined tip x (gradient method)


class CrackTracker:
    """
    Tracks delamination crack tip and computes propagation length.

    Parameters
    ----------
    min_gap_pixels : int
        Minimum boundary separation to count a column as 'cracked'.
    smooth_window : int
        Number of recent frames used for the rolling-median tip smoother.
    crack_direction : str
        'right'  → crack propagates toward higher x (default DCB geometry).
        'left'   → crack propagates toward lower x.
    """

    def __init__(
        self,
        min_gap_pixels: int = 5,
        smooth_window: int = 7,
        crack_direction: str = "right",
    ):
        if crack_direction not in ("right", "left"):
            raise ValueError("crack_direction must be 'right' or 'left'")

        self.min_gap = min_gap_pixels
        self.smooth_window = smooth_window
        self.crack_direction = crack_direction

        self._tip_x_hist: deque[int] = deque(maxlen=smooth_window)
        self._tip_y_hist: deque[int] = deque(maxlen=smooth_window)

        self._initial_tip: Optional[tuple[int, int]] = None
        self._peak_tip_x: Optional[int] = None   # integer monotonicity guard
        self._peak_sp_x: Optional[float] = None  # subpixel monotonicity guard
        self.states: list[FrameState] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_index: int,
        upper_pts: Optional[np.ndarray],
        lower_pts: Optional[np.ndarray],
        subpixel_tip_x: Optional[float] = None,
    ) -> FrameState:
        """
        Update tracker with newly detected boundaries and return a FrameState.

        *upper_pts* and *lower_pts* are Nx2 arrays (x, y) in ROI-local coords,
        or None when detection failed.
        """
        raw_tip, gap = self._find_raw_tip(upper_pts, lower_pts)

        if raw_tip is not None:
            self._tip_x_hist.append(raw_tip[0])
            self._tip_y_hist.append(raw_tip[1])
            # Track the furthest tip seen (crack can't propagate backward)
            if self._peak_tip_x is None:
                self._peak_tip_x = raw_tip[0]
            elif self.crack_direction == "right":
                self._peak_tip_x = max(self._peak_tip_x, raw_tip[0])
            else:
                self._peak_tip_x = min(self._peak_tip_x, raw_tip[0])

        smoothed = self._smoothed_tip()

        # Latch the initial tip once we have a stable first estimate
        if self._initial_tip is None and smoothed is not None:
            if len(self._tip_x_hist) >= min(3, self.smooth_window):
                self._initial_tip = smoothed

        if smoothed is not None:
            sx, sy = smoothed
            # Monotonicity guard: don't let the smoothed tip retreat more than
            # a few pixels behind the recorded peak (jitter allowed, reversals
            # are not — cracks don't heal).
            if self._peak_tip_x is not None:
                if self.crack_direction == "right" and sx < self._peak_tip_x - 8:
                    sx = self._peak_tip_x
                elif self.crack_direction == "left" and sx > self._peak_tip_x + 8:
                    sx = self._peak_tip_x
            tip_x, tip_y = sx, sy
        else:
            tip_x, tip_y = 0, 0

        # Sub-pixel tip x: use gradient-refined value when available and
        # monotonicity-consistent; otherwise fall back to integer tip_x.
        sp_x: float = float(tip_x)
        if subpixel_tip_x is not None and tip_x > 0:
            if self.crack_direction == "right":
                if subpixel_tip_x >= tip_x - 1.0:
                    sp_x = max(float(tip_x), subpixel_tip_x)
            else:
                if subpixel_tip_x <= tip_x + 1.0:
                    sp_x = min(float(tip_x), subpixel_tip_x)

        # Subpixel monotonicity guard — crack cannot retreat more than 5px.
        if self._peak_sp_x is None:
            self._peak_sp_x = sp_x
        elif self.crack_direction == "right":
            if sp_x < self._peak_sp_x - 5.0:
                sp_x = self._peak_sp_x
            else:
                self._peak_sp_x = max(self._peak_sp_x, sp_x)
        else:
            if sp_x > self._peak_sp_x + 5.0:
                sp_x = self._peak_sp_x
            else:
                self._peak_sp_x = min(self._peak_sp_x, sp_x)

        prop_px, prop_h_px = self._propagation_length_subpixel((sp_x, float(tip_y)))

        state = FrameState(
            frame_index=frame_index,
            tip_x=tip_x,
            tip_y=tip_y,
            upper_boundary=upper_pts,
            lower_boundary=lower_pts,
            propagation_px=prop_px,
            propagation_h_px=prop_h_px,
            gap_at_tip=gap,
            tip_x_subpixel=sp_x,
        )
        self.states.append(state)
        return state

    @property
    def initial_tip(self) -> Optional[tuple[int, int]]:
        return self._initial_tip

    def reset(self):
        """Clear all state for reprocessing."""
        self._tip_x_hist.clear()
        self._tip_y_hist.clear()
        self._initial_tip = None
        self._peak_tip_x = None
        self._peak_sp_x = None
        self.states.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_raw_tip(
        self,
        upper_pts: Optional[np.ndarray],
        lower_pts: Optional[np.ndarray],
    ) -> tuple[Optional[tuple[int, int]], int]:
        """
        Scan columns where both boundaries are defined and find the extremal
        column with a gap >= min_gap.  Returns (tip_xy, gap_width_at_tip).
        """
        if upper_pts is None or lower_pts is None:
            return None, 0

        # Build fast lookup dicts: x → y for each boundary
        u_dict = {int(p[0]): int(p[1]) for p in upper_pts}
        l_dict = {int(p[0]): int(p[1]) for p in lower_pts}

        common_xs = sorted(set(u_dict) & set(l_dict))
        if not common_xs:
            return None, 0

        # Filter to columns with a meaningful gap
        cracked = [(x, l_dict[x] - u_dict[x]) for x in common_xs
                   if l_dict[x] - u_dict[x] >= self.min_gap]

        if not cracked:
            return None, 0

        # Extremal cracked column = crack tip
        if self.crack_direction == "right":
            tip_x, gap = max(cracked, key=lambda t: t[0])
        else:
            tip_x, gap = min(cracked, key=lambda t: t[0])

        tip_y = (u_dict[tip_x] + l_dict[tip_x]) // 2
        return (tip_x, tip_y), gap

    def _smoothed_tip(self) -> Optional[tuple[int, int]]:
        if not self._tip_x_hist:
            return None
        return (int(np.median(self._tip_x_hist)), int(np.median(self._tip_y_hist)))

    def _propagation_length(
        self, current_tip: tuple[int, int]
    ) -> tuple[float, float]:
        if self._initial_tip is None or current_tip == (0, 0):
            return 0.0, 0.0
        dx = current_tip[0] - self._initial_tip[0]
        dy = current_tip[1] - self._initial_tip[1]
        euclidean = float(np.sqrt(dx ** 2 + dy ** 2))
        horizontal = float(abs(dx))
        return euclidean, horizontal

    def _propagation_length_subpixel(
        self, current_tip: tuple[float, float]
    ) -> tuple[float, float]:
        """Like _propagation_length but accepts float tip coordinates."""
        if self._initial_tip is None or current_tip == (0.0, 0.0):
            return 0.0, 0.0
        dx = current_tip[0] - self._initial_tip[0]
        dy = current_tip[1] - self._initial_tip[1]
        return float(np.sqrt(dx ** 2 + dy ** 2)), float(abs(dx))
