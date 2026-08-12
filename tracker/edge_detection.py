"""
edge_detection.py
-----------------
Detects the upper and lower delamination crack boundaries within an ROI.

Two detection modes are available; the right one is chosen automatically:

  'threshold'  (default for high-contrast videos)
  ─────────────────────────────────────────────────
  Best when: bright specimen (GFRP/CFRP arms) on a dark background.
  Method:
    1. CLAHE + Gaussian blur for mild normalisation.
    2. Otsu binary threshold → white regions = arms, black = background/gap.
    3. Column-by-column scan: find all connected white runs.
       • If ≥2 separate white runs → cracked column (two arms visible).
         Upper boundary = bottom pixel of the upper run (edge of upper arm).
         Lower boundary = top pixel of the lower run  (edge of lower arm).
       • If only 1 run → intact column (arms still bonded).
    4. Polynomial fit + IQR outlier removal for smooth boundaries.

  'canny'  (fallback for moderate-contrast or noisy videos)
  ──────────────────────────────────────────────────────────
  Best when: crack appears as a dark or bright gap with moderate contrast.
  Method: CLAHE → Gaussian blur → Canny → horizontal dilation →
          column scan with arm/gap contrast gate → polynomial fit.

  Auto-detection heuristic
  ─────────────────────────
  If >35 % of ROI pixels are very dark (<60) AND >10 % are very bright
  (>150), the image is clearly bimodal (specimen vs background) and
  'threshold' mode is selected.  Otherwise 'canny' is used.
"""

from __future__ import annotations

import cv2
import numpy as np


def _smooth1d(arr: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Gaussian-smooth a 1-D float array without scipy dependency."""
    r = max(1, int(np.ceil(2.5 * sigma)))
    t = np.arange(-r, r + 1, dtype=float)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode='same')


class EdgeDetector:
    """
    Extracts upper and lower delamination boundaries from a grayscale ROI.

    Parameters
    ----------
    min_gap_pixels : int
        Minimum vertical gap (pixels) between the two boundaries for a
        column to count as cracked.
    min_run_pixels : int
        Minimum height of a white pixel run to be considered an arm
        (rejects tiny noise blobs in threshold mode).
    clahe_clip : float
        CLAHE clip limit.
    blur_kernel : int
        Gaussian blur kernel size (forced odd).
    poly_degree : int
        Polynomial degree for boundary smoothing.
    dilate_iters : int
        Horizontal dilation iterations (Canny mode only).
    min_arm_gap_contrast : float
        Arm-to-gap contrast fraction of 0–255 (Canny mode only).
    bright_crack : bool
        True  → reflective crack (gap brighter than arms).
        False → shadow crack   (gap darker  than arms).  [Canny mode only]
    mode : str
        'auto' | 'threshold' | 'canny'
    """

    def __init__(
        self,
        min_gap_pixels: int = 3,
        min_run_pixels: int = 12,
        min_threshold: int = 110,
        clahe_clip: float = 1.5,
        blur_kernel: int = 3,
        poly_degree: int = 2,
        dilate_iters: int = 2,
        min_arm_gap_contrast: float = 0.06,
        min_arm_brightness: float = 145.0,
        min_lower_brightness: float = 130.0,
        max_gap_search: int = 65,
        max_gap_pixels: int = 90,
        max_lower_y_frac: float = 0.82,
        bright_crack: bool = False,
        mode: str = "auto",
    ):
        bk = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.min_gap = min_gap_pixels
        self.min_run = min_run_pixels
        # Minimum absolute threshold applied after Otsu.
        # Prevents grey overlays / watermarks (≈ 60–110 intensity) from
        # being misclassified as specimen material.
        self.min_threshold = min_threshold
        self.poly_degree = poly_degree
        self.dilate_iters = dilate_iters
        self.min_arm_gap_contrast = min_arm_gap_contrast
        # Minimum mean intensity (CLAHE image) for the UPPER arm run.
        # GFRP upper arm ≈ 166-230; watermark ≈ 114-120 → 145 separates them.
        self.min_arm_brightness = min_arm_brightness
        # Lower threshold for the LOWER arm: it can be dimmer because it
        # partially overlaps the watermark zone in later frames.
        self.min_lower_brightness = min_lower_brightness
        # How far below the upper arm bottom we search for the lower arm.
        self.max_gap_search = max_gap_search
        # Maximum allowed median gap (px) between fitted upper and lower
        # boundaries.  Detections exceeding this are rejected as false positives.
        self.max_gap_pixels = max_gap_pixels
        # Lower arm start must be above this fraction of ROI height.
        # Prevents the watermark (global y≈295-320) and lower fixture from
        # being identified as the lower arm boundary.
        self.max_lower_y_frac = max_lower_y_frac
        self.bright_crack = bright_crack
        self.mode = mode

        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        self._blur_size = (bk, bk)
        self._h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))

        self._prev_upper: np.ndarray | None = None
        self._prev_lower: np.ndarray | None = None
        # Cumulative x→y maps.  Once a column is detected it stays in the
        # map and its y-value is refreshed whenever that column is detected
        # again.  This prevents markers from disappearing as detection range
        # shifts, while still updating positions with fresh measurements.
        self._cumul_upper: dict[int, int] = {}
        self._cumul_lower: dict[int, int] = {}
        # Columns permanently excluded: once an arm exits the ROI (cumul entry
        # deleted because new_y is out of [0, h_roi)), the x-column is added
        # here and the consistency filter always rejects detections there.
        self._deleted_xs: set[int] = set()
        self._active_mode: str = "threshold"   # resolved after first frame
        self._subpixel_tip_x: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, roi: np.ndarray) -> np.ndarray:
        enhanced = self._clahe.apply(roi)
        return cv2.GaussianBlur(enhanced, self._blur_size, 0)

    def detect(
        self, roi: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Detect upper and lower delamination boundaries in a grayscale ROI.

        Returns Nx2 (x, y) arrays in ROI-local coordinates built from the
        cumulative boundary map — markers never disappear once a column has
        been detected.  Falls back to _prev or None when nothing detected yet.
        """
        preprocessed = self.preprocess(roi)

        if self.mode == "auto":
            self._active_mode = self._select_mode(roi)
        else:
            self._active_mode = self.mode

        # Build guided-scan dict from current cumul state.  Already-tracked
        # columns will be searched in a small window around their last known
        # arm position instead of the normal top-down scan, so detection
        # succeeds even when the inter-arm gap exceeds max_gap_search.
        guided: dict[int, tuple[int, int]] = {
            xi: (self._cumul_upper[xi], self._cumul_lower[xi])
            for xi in self._cumul_upper
            if xi in self._cumul_lower
        }

        # Primary detection: binary column scan.
        # _scan_binary returns a third value: the rightmost x detected by the
        # STANDARD (non-guided) branch.  This is the true crack-tip x without
        # any cumul-guidance extension, used to cap the cumul update below.
        if self._active_mode == "threshold":
            raw_upper, raw_lower, _x_std_binary = self._scan_binary(
                preprocessed, guided=guided or None
            )
        else:
            edge_map = self._build_edge_map(preprocessed)
            raw_upper, raw_lower = self._scan_canny(edge_map, preprocessed)
            _x_std_binary = -1

        # (Connected-components gap-fill removed: caused oscillating wrong
        # boundary positions in newly-detected regions after stick-slip events,
        # making the blue overlay unstable after large crack jumps.)

        # Consistency + permanent-exclusion filter.
        # Reject a detected column pair if:
        #   (a) the column previously exited the ROI (_deleted_xs), OR
        #   (b) either arm deviates >_CTOL px from its last known position.
        # (a) prevents grip / fixture false-detections from entering after the
        #     real arm has permanently left the ROI.
        # (b) suppresses sudden false-detections near the ROI boundary while
        #     the arm is still inside.
        _CTOL = 30
        if (self._cumul_upper or self._deleted_xs) and raw_upper:
            keep_u: list = []
            keep_l: list = []
            for (xu, yu), (xl, yl) in zip(raw_upper, raw_lower):
                xi = int(xu)
                if xi in self._deleted_xs:
                    continue  # column permanently excluded
                u_ok = (xi not in self._cumul_upper
                        or abs(int(yu) - self._cumul_upper[xi]) <= _CTOL)
                l_ok = (xi not in self._cumul_lower
                        or abs(int(yl) - self._cumul_lower[xi]) <= _CTOL)
                if u_ok and l_ok:
                    keep_u.append((xu, yu))
                    keep_l.append((xl, yl))
            raw_upper, raw_lower = keep_u, keep_l

        def _from_cumul():
            """Return Nx2 arrays from cumulative dicts."""
            if not self._cumul_upper:
                return self._prev_upper, self._prev_lower

            # Only include columns with a positive raw gap.  Median smoothing
            # can bleed gap-0 neighbours into the output, which creates false
            # detections in the fixture area and shifts the initial-tip anchor.
            xs = sorted(
                k for k in self._cumul_upper
                if k in self._cumul_lower
                and self._cumul_upper[k] < self._cumul_lower[k]
            )
            if not xs:
                return self._prev_upper, self._prev_lower

            # Stored positions are at the arm/gap interface (anchor+1 / anchor-1)
            # from the binary scan.  Apply a ±3-column sliding median to suppress
            # individual-column outliers (missed anchor, short-run instability)
            # without distorting the smooth arm curvature.
            n = len(xs)
            yu_raw = [self._cumul_upper[xi] for xi in xs]
            yl_raw = [self._cumul_lower[xi] for xi in xs]
            _WIN = 3

            out_u: list[tuple[int, int]] = []
            out_l: list[tuple[int, int]] = []

            for i, xi in enumerate(xs):
                i0 = max(0, i - _WIN)
                i1 = min(n, i + _WIN + 1)
                yu = int(np.median(yu_raw[i0:i1]))
                yl = int(np.median(yl_raw[i0:i1]))

                if yu >= yl:
                    continue

                out_u.append((xi, yu))
                out_l.append((xi, yl))

            if not out_u:
                return self._prev_upper, self._prev_lower
            return (
                np.array(out_u, dtype=float),
                np.array(out_l, dtype=float),
            )

        min_pts = max(self.poly_degree + 1, 4)
        if len(raw_upper) < min_pts:
            return _from_cumul()

        upper = self._fit_boundary(np.array(raw_upper, dtype=float))
        lower = self._fit_boundary(np.array(raw_lower, dtype=float))

        # Local-fit window: only the last N points are used for polynomial
        # extrapolation in every extension scan.  Using all points causes
        # the global arm curvature to dominate the prediction, so the
        # extrapolated tip region drifts far from the actual arm edge.
        _FIT_TAIL = 60

        # Arm brightness gate for extension-scan seeds.
        # If the predicted upper-arm position has no bright pixels nearby,
        # the linear extrapolation has drifted outside the arm — stop.
        _ARM_BRIGHT = 55  # minimum CLAHE value to count as "arm material"

        def _seed_ok(pre: np.ndarray, ey: int, ly: int, x: int) -> bool:
            """True only when predicted arm positions land on bright material."""
            h = pre.shape[0]
            # Check a small window 2-4 px above the predicted inner edge.
            u_check = max(0, ey - 3)
            l_check = min(h - 1, ly + 3)
            return (int(pre[u_check, x]) >= _ARM_BRIGHT
                    and int(pre[l_check, x]) >= _ARM_BRIGHT)

        if upper is not None and lower is not None:
            # Near-tip lookahead: once the crack is established, try to push
            # detection a little further toward the true tip using linear
            # extrapolation as a guided seed and a relaxed min_run.
            if len(self._cumul_upper) >= 30:
                x_det_max = int(upper[-1, 0])
                h_pre, w_pre = preprocessed.shape
                ext_end = min(w_pre, x_det_max + 40)
                if x_det_max + 1 < ext_end:
                    try:
                        _up_e = upper[-_FIT_TAIL:] if len(upper) > _FIT_TAIL else upper
                        _lo_e = lower[-_FIT_TAIL:] if len(lower) > _FIT_TAIL else lower
                        cu = np.polyfit(_up_e[:, 0], _up_e[:, 1],
                                        min(1, len(_up_e) - 1))
                        cl = np.polyfit(_lo_e[:, 0], _lo_e[:, 1],
                                        min(1, len(_lo_e) - 1))
                        seeds_ext: dict[int, tuple[int, int]] = {}
                        _last_ey_e = _last_ly_e = None
                        for x in range(x_det_max + 1, ext_end):
                            if x in self._deleted_xs:
                                continue
                            ey = int(np.clip(np.polyval(cu, x), 0, h_pre - 1))
                            ly = int(np.clip(np.polyval(cl, x), 0, h_pre - 1))
                            if not _seed_ok(preprocessed, ey, ly, x):
                                break
                            if ly > ey:
                                _last_ey_e, _last_ly_e = ey, ly
                                seeds_ext[x] = (ey, ly)
                            elif _last_ey_e is not None:
                                seeds_ext[x] = (_last_ey_e, _last_ly_e)
                        if seeds_ext:
                            ext_u, ext_l = self._scan_tip_extension(
                                preprocessed, seeds_ext
                            )
                            if ext_u:
                                raw_upper = raw_upper + ext_u
                                raw_lower = raw_lower + ext_l
                                upper = self._fit_boundary(
                                    np.array(raw_upper, dtype=float)
                                )
                                lower = self._fit_boundary(
                                    np.array(raw_lower, dtype=float)
                                )
                    except Exception:
                        pass

            # Gradient-based sub-pixel extension.
            self._subpixel_tip_x = None
            if upper is not None and lower is not None:
                x_gstart = int(upper[-1, 0]) + 1
                h_g, w_g = preprocessed.shape
                ext_end_g = min(w_g, x_gstart + 30)
                if x_gstart < ext_end_g:
                    try:
                        _up_g = upper[-_FIT_TAIL:] if len(upper) > _FIT_TAIL else upper
                        _lo_g = lower[-_FIT_TAIL:] if len(lower) > _FIT_TAIL else lower
                        cu_g = np.polyfit(
                            _up_g[:, 0], _up_g[:, 1],
                            min(1, len(_up_g) - 1),
                        )
                        cl_g = np.polyfit(
                            _lo_g[:, 0], _lo_g[:, 1],
                            min(1, len(_lo_g) - 1),
                        )
                        seeds_grad: dict[int, tuple[int, int]] = {}
                        _last_ey_g = _last_ly_g = None
                        for x in range(x_gstart, ext_end_g):
                            if x in self._deleted_xs:
                                continue
                            ey = int(np.clip(np.polyval(cu_g, x), 0, h_g - 1))
                            ly = int(np.clip(np.polyval(cl_g, x), 0, h_g - 1))
                            if not _seed_ok(preprocessed, ey, ly, x):
                                break
                            if ly > ey:
                                _last_ey_g, _last_ly_g = ey, ly
                                seeds_grad[x] = (ey, ly)
                            elif _last_ey_g is not None:
                                seeds_grad[x] = (_last_ey_g, _last_ly_g)
                        if seeds_grad:
                            grad_u, grad_l, sp_x = self._scan_tip_gradient(
                                preprocessed, seeds_grad
                            )
                            if grad_u:
                                raw_upper = raw_upper + grad_u
                                raw_lower = raw_lower + grad_l
                                upper = self._fit_boundary(
                                    np.array(raw_upper, dtype=float)
                                )
                                lower = self._fit_boundary(
                                    np.array(raw_lower, dtype=float)
                                )
                                self._subpixel_tip_x = sp_x
                    except Exception:
                        pass

            # Valley-based sub-pixel extension.
            if upper is not None and lower is not None:
                x_vstart = int(upper[-1, 0]) + 1
                h_v, w_v = preprocessed.shape
                ext_end_v = min(w_v, x_vstart + 35)
                if x_vstart < ext_end_v:
                    try:
                        _up_v = upper[-_FIT_TAIL:] if len(upper) > _FIT_TAIL else upper
                        _lo_v = lower[-_FIT_TAIL:] if len(lower) > _FIT_TAIL else lower
                        cu_v = np.polyfit(
                            _up_v[:, 0], _up_v[:, 1],
                            min(1, len(_up_v) - 1),
                        )
                        cl_v = np.polyfit(
                            _lo_v[:, 0], _lo_v[:, 1],
                            min(1, len(_lo_v) - 1),
                        )
                        seeds_val: dict[int, tuple[int, int]] = {}
                        _last_ey_v = _last_ly_v = None
                        for x in range(x_vstart, ext_end_v):
                            if x in self._deleted_xs:
                                continue
                            ey = int(np.clip(np.polyval(cu_v, x), 0, h_v - 1))
                            ly = int(np.clip(np.polyval(cl_v, x), 0, h_v - 1))
                            if not _seed_ok(preprocessed, ey, ly, x):
                                break
                            if ly > ey:
                                _last_ey_v, _last_ly_v = ey, ly
                                seeds_val[x] = (ey, ly)
                            elif _last_ey_v is not None:
                                seeds_val[x] = (_last_ey_v, _last_ly_v)
                        if seeds_val:
                            val_u, val_l, sp_x_v = self._scan_tip_valley(
                                preprocessed, seeds_val
                            )
                            if val_u:
                                raw_upper = raw_upper + val_u
                                raw_lower = raw_lower + val_l
                                upper = self._fit_boundary(
                                    np.array(raw_upper, dtype=float)
                                )
                                lower = self._fit_boundary(
                                    np.array(raw_lower, dtype=float)
                                )
                                if sp_x_v is not None and (
                                    self._subpixel_tip_x is None
                                    or sp_x_v > self._subpixel_tip_x
                                ):
                                    self._subpixel_tip_x = sp_x_v
                    except Exception:
                        pass

            # Dark-pixel scan — 4th extension layer.
            if upper is not None and lower is not None and len(upper) >= 30:
                x_dstart = int(upper[-1, 0]) + 1
                h_d, w_d = preprocessed.shape
                ext_end_d = min(w_d, x_dstart + 20)
                if x_dstart < ext_end_d:
                    try:
                        _up_d = upper[-_FIT_TAIL:] if len(upper) > _FIT_TAIL else upper
                        _lo_d = lower[-_FIT_TAIL:] if len(lower) > _FIT_TAIL else lower
                        cu_d = np.polyfit(
                            _up_d[:, 0], _up_d[:, 1],
                            min(1, len(_up_d) - 1),
                        )
                        cl_d = np.polyfit(
                            _lo_d[:, 0], _lo_d[:, 1],
                            min(1, len(_lo_d) - 1),
                        )
                        seeds_dark: dict[int, tuple[int, int]] = {}
                        _last_ey_d = _last_ly_d = None
                        for x in range(x_dstart, ext_end_d):
                            if x in self._deleted_xs:
                                continue
                            ey = int(np.clip(np.polyval(cu_d, x), 0, h_d - 1))
                            ly = int(np.clip(np.polyval(cl_d, x), 0, h_d - 1))
                            if not _seed_ok(preprocessed, ey, ly, x):
                                break
                            if ly > ey:
                                _last_ey_d, _last_ly_d = ey, ly
                                seeds_dark[x] = (ey, ly)
                            elif _last_ey_d is not None:
                                seeds_dark[x] = (_last_ey_d, _last_ly_d)
                        if seeds_dark:
                            dark_u, dark_l, sp_x_d = self._scan_tip_dark(
                                preprocessed, seeds_dark
                            )
                            if dark_u:
                                raw_upper = raw_upper + dark_u
                                raw_lower = raw_lower + dark_l
                                upper = self._fit_boundary(
                                    np.array(raw_upper, dtype=float)
                                )
                                lower = self._fit_boundary(
                                    np.array(raw_lower, dtype=float)
                                )
                                if sp_x_d is not None and (
                                    self._subpixel_tip_x is None
                                    or sp_x_d > self._subpixel_tip_x
                                ):
                                    self._subpixel_tip_x = sp_x_d
                    except Exception:
                        pass

            if upper is None or lower is None:
                return _from_cumul()

            # Sanity check on the rightmost columns only — the left (wide-open)
            # region has a legitimately large arm-to-arm gap that would
            # incorrectly trigger the global median guard.
            tail = min(100, len(upper))
            median_gap = float(
                np.median(lower[-tail:, 1]) - np.median(upper[-tail:, 1])
            )
            if median_gap > self.max_gap_pixels:
                return _from_cumul()

            # Update cumulative map with raw detected positions only.
            # Using polynomial values here caused wrong y positions to persist
            # across frames when the arm curvature exceeded what a degree-2
            # polynomial could represent.  Raw points (binary scan + extensions)
            # are the actual arm-edge measurements — the polynomial is only
            # needed as a seed for the extension look-ahead, not for storage.
            # Gap guard: reject any pair where upper_y >= lower_y (impossible).
            #
            # EMA smoothing (α=0.5): blends new raw measurement with stored
            # value to suppress frame-to-frame jitter without lagging behind
            # the slow DCB arm opening.
            # Adaptive EMA: small genuine movement → α=0.7 (responsive, low lag).
            # Large jump → α=0.3 (dampened, suppresses noisy outlier detections).
            _ALPHA_SMALL = 0.7   # change ≤ _GATE px
            _ALPHA_LARGE = 0.3   # change  > _GATE px
            _GATE        = 6     # pixel threshold separating the two regimes
            _MAX_EXT_CUMUL = 15
            if _x_std_binary >= 0:
                _x_cap = _x_std_binary + _MAX_EXT_CUMUL
            elif self._cumul_upper:
                _x_cap = max(self._cumul_upper.keys())
            else:
                _x_cap = int(1e9)
            _raw_u_dict = {int(x): int(y) for x, y in raw_upper
                           if int(x) <= _x_cap}
            _raw_l_dict = {int(x): int(y) for x, y in raw_lower
                           if int(x) <= _x_cap}
            for xi, yu in _raw_u_dict.items():
                yl = _raw_l_dict.get(xi)
                if yl is None or yu < yl:
                    if xi in self._cumul_upper:
                        delta = abs(yu - self._cumul_upper[xi])
                        a = _ALPHA_SMALL if delta <= _GATE else _ALPHA_LARGE
                        yu = int(round((1.0 - a) * self._cumul_upper[xi] + a * yu))
                    self._cumul_upper[xi] = yu
            for xi, yl in _raw_l_dict.items():
                yu = _raw_u_dict.get(xi)
                if yu is None or yl > yu:
                    if xi in self._cumul_lower:
                        delta = abs(yl - self._cumul_lower[xi])
                        a = _ALPHA_SMALL if delta <= _GATE else _ALPHA_LARGE
                        yl = int(round((1.0 - a) * self._cumul_lower[xi] + a * yl))
                    self._cumul_lower[xi] = yl
            # Post-EMA cross-check: ensure upper < lower after blending.
            for xi in list(self._cumul_upper):
                if xi in self._cumul_lower and self._cumul_upper[xi] >= self._cumul_lower[xi]:
                    self._cumul_upper[xi] = self._cumul_lower[xi] - 1

            # Remove columns where either arm has reached the ROI edge.
            # Prevents the upper arm from locking at the ROI top and the
            # lower arm from locking onto the grip/fixture at the ROI bottom.
            _EDGE = 5
            h_roi = roi.shape[0]
            to_del: set[int] = set()
            for xi, yu in self._cumul_upper.items():
                if yu <= _EDGE:
                    to_del.add(xi)
            for xi, yl in self._cumul_lower.items():
                if yl >= h_roi - _EDGE:
                    to_del.add(xi)
            for xi in to_del:
                self._cumul_upper.pop(xi, None)
                self._cumul_lower.pop(xi, None)
                self._deleted_xs.add(xi)

            # Keep a simple "last good fit" as fallback for total detection failure.
            x_span = int(upper[-1, 0]) - int(upper[0, 0])
            if x_span >= 60:
                self._prev_upper = upper
                self._prev_lower = lower

            return _from_cumul()

        return _from_cumul()

    def reset_state(self):
        self._prev_upper = None
        self._prev_lower = None
        self._cumul_upper.clear()
        self._cumul_lower.clear()
        self._deleted_xs.clear()
        self._subpixel_tip_x = None

    @property
    def active_mode(self) -> str:
        return self._active_mode

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_mode(roi: np.ndarray) -> str:
        """Bimodal histogram → threshold mode; otherwise canny."""
        flat = roi.flatten().astype(float)
        dark_frac   = float(np.mean(flat < 60))
        bright_frac = float(np.mean(flat > 150))
        if dark_frac > 0.35 and bright_frac > 0.10:
            return "threshold"
        return "canny"

    # ------------------------------------------------------------------
    # Connected-component primary detection
    # ------------------------------------------------------------------

    def _detect_by_components(
        self, preprocessed: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        İki GFRP kolunu bağlı bileşen olarak bulur ve her sütun için
        içteki sınırları döndürür:
          upper_raw: (x, y) → üst kolun alt pikseli (çatlak üst kenarı)
          lower_raw: (x, y) → alt kolun üst pikseli (çatlak alt kenarı)

        İki ayrı bileşen bulunamazsa boş liste döner; çağıran fallback
        olarak binary scan'e geçer.
        """
        h, w = preprocessed.shape

        otsu_val, _ = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thresh = max(int(otsu_val), self.min_threshold)
        _, binary = cv2.threshold(preprocessed, thresh, 255, cv2.THRESH_BINARY)

        # Dikey gürültüyü temizle
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if n_labels < 3:
            return [], []

        # Kol adayları: yeterli alan ve yüksekliğe sahip bileşenler
        _MIN_AREA   = 400
        _MIN_HEIGHT = max(8, int(h * 0.10))

        candidates: list[tuple[int, int, float]] = []  # (label, area, cy)
        for i in range(1, n_labels):
            area   = int(stats[i, cv2.CC_STAT_AREA])
            height = int(stats[i, cv2.CC_STAT_HEIGHT])
            cy     = float(centroids[i][1])
            if area >= _MIN_AREA and height >= _MIN_HEIGHT:
                candidates.append((i, area, cy))

        if len(candidates) < 2:
            return [], []

        # En büyük iki bileşen = iki kol
        candidates.sort(key=lambda c: -c[1])
        c1, c2 = candidates[0], candidates[1]

        # Üst kol: daha küçük merkez y
        if c1[2] < c2[2]:
            upper_label, lower_label = c1[0], c2[0]
        else:
            upper_label, lower_label = c2[0], c1[0]

        upper_raw: list[tuple[int, int]] = []
        lower_raw: list[tuple[int, int]] = []

        for x in range(w):
            col = labels[:, x]

            # Üst kol: en alt piksel (iç kenar = çatlak üstü)
            u_ys = np.where(col == upper_label)[0]
            if len(u_ys) >= 1:
                upper_raw.append((x, int(u_ys[-1])))

            # Alt kol: en üst piksel (iç kenar = çatlak altı)
            l_ys = np.where(col == lower_label)[0]
            if len(l_ys) >= 1:
                lower_raw.append((x, int(l_ys[0])))

        return upper_raw, lower_raw

    # ------------------------------------------------------------------
    # Threshold mode
    # ------------------------------------------------------------------

    def _scan_binary(
        self, preprocessed: np.ndarray, guided: dict | None = None
    ) -> tuple[list, list, int]:
        """
        Otsu-threshold the ROI, then for each column find connected white
        runs.  Two or more distinct runs → the column is cracked.

        For columns already tracked (present in *guided*), a ±_GUIDE px
        window search around the last known arm position replaces the normal
        top-down scan.  This lets the detector find both arms even when the
        inter-arm gap is much larger than max_gap_search (wide-open region
        near the loading point at large crack lengths).
        """
        otsu_val, _ = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thresh = max(int(otsu_val), self.min_threshold)
        _, binary = cv2.threshold(preprocessed, thresh, 255, cv2.THRESH_BINARY)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

        h, w = binary.shape
        upper_raw: list[tuple[int, int]] = []
        lower_raw: list[tuple[int, int]] = []
        std_max_x: int = -1   # rightmost x detected by STANDARD (non-guided) branch
        _GUIDE = 25  # guided-search half-window (pixels)

        for x in range(w):
            col   = binary[:, x]
            col_g = preprocessed[:, x].astype(float)

            if guided and x in guided:
                # ── Guided scan ──────────────────────────────────────────
                # Both arms are sought independently near their expected loci.
                # No max_gap_search / max_lower_y_frac limits apply here —
                # the guided position already encodes a reliable prior.
                exp_uy, exp_ly = guided[x]

                # Upper arm window
                y0u = max(0, exp_uy - _GUIDE)
                y1u = min(h - 1, exp_uy + _GUIDE)
                seg_u  = col[y0u : y1u + 1]
                runs_u = self._white_runs(seg_u, len(seg_u))
                valid_u = [(s + y0u, e + y0u) for s, e in runs_u
                           if (e - s + 1) >= self.min_run]

                upper_run = None
                for s, e in valid_u:
                    if float(np.mean(col_g[s : e + 1])) >= self.min_arm_brightness:
                        upper_run = (s, e)
                        break
                if upper_run is None:
                    continue

                # Stable anchor: last pixel ≥ _EDGE_T in the upper arm run.
                # _EDGE_T=120 is above the Otsu variation range (≈110-140) so
                # the anchor lands inside the clearly-white arm and does NOT
                # shift when Otsu drifts — prevents the blue marker from
                # oscillating between white and gray regions.
                # Report one pixel BELOW the anchor (grey interface pixel).
                _EDGE_T = 120
                top_anchor = upper_run[1]  # fallback: Otsu boundary
                for yi in range(upper_run[0], upper_run[1] + 1):
                    if col_g[yi] >= _EDGE_T:
                        top_anchor = yi
                top_edge = min(top_anchor + 1, h - 1)

                # Lower arm window (strictly below upper arm)
                y0l = max(top_edge + 1, exp_ly - _GUIDE)
                y1l = min(h - 1, exp_ly + _GUIDE)
                if y0l > y1l:
                    continue
                seg_l  = col[y0l : y1l + 1]
                runs_l = self._white_runs(seg_l, len(seg_l))
                valid_l = [(s + y0l, e + y0l) for s, e in runs_l
                           if (e - s + 1) >= self.min_run]

                lower_run = None
                for s, e in valid_l:
                    if float(np.mean(col_g[s : e + 1])) >= self.min_lower_brightness:
                        lower_run = (s, e)
                        break
                if lower_run is None:
                    continue

                # Stable anchor for lower arm: first pixel ≥ _EDGE_T.
                # Report one pixel ABOVE the anchor (grey interface pixel).
                bot_anchor = lower_run[0]  # fallback: Otsu boundary
                for yi in range(lower_run[0], lower_run[1] + 1):
                    if col_g[yi] >= _EDGE_T:
                        bot_anchor = yi
                        break
                bot_edge = max(0, bot_anchor - 1)
                if (bot_edge - top_edge) < self.min_gap:
                    continue

                upper_raw.append((x, top_edge))
                lower_raw.append((x, bot_edge))

            else:
                # ── Standard scan (existing logic) ───────────────────────
                runs  = self._white_runs(col, h)
                valid = [(s, e) for s, e in runs if (e - s + 1) >= self.min_run]

                upper_run = None
                for s, e in valid:
                    if float(np.mean(col_g[s : e + 1])) >= self.min_arm_brightness:
                        upper_run = (s, e)
                        break
                if upper_run is None:
                    continue

                # Stable anchor: last pixel ≥ _EDGE_T in the upper run.
                # _EDGE_T=120 > Otsu variation range → anchor stays inside
                # clearly-white arm regardless of per-frame Otsu shift.
                _EDGE_T = 120
                top_anchor = upper_run[1]  # fallback: Otsu boundary
                for yi in range(upper_run[0], upper_run[1] + 1):
                    if col_g[yi] >= _EDGE_T:
                        top_anchor = yi
                top_edge = min(top_anchor + 1, h - 1)
                search_limit = min(h - 1, top_anchor + self.max_gap_search)

                lower_run = None
                for s, e in valid:
                    if s <= upper_run[1]:
                        continue
                    if s > search_limit:
                        break
                    clip_e   = min(e, search_limit)
                    seg_bright = float(np.mean(col_g[s : clip_e + 1]))
                    if seg_bright >= self.min_lower_brightness:
                        lower_run = (s, clip_e)
                        break
                if lower_run is None:
                    continue

                if lower_run[0] > int(h * self.max_lower_y_frac):
                    continue

                # Stable anchor for lower run: first pixel ≥ _EDGE_T.
                # Report one pixel ABOVE the anchor (arm/gap interface).
                bot_anchor = lower_run[0]  # fallback: Otsu boundary
                for yi in range(lower_run[0], lower_run[1] + 1):
                    if col_g[yi] >= _EDGE_T:
                        bot_anchor = yi
                        break
                bot_edge = max(0, bot_anchor - 1)
                if (bot_edge - top_edge) < self.min_gap:
                    continue

                upper_raw.append((x, top_edge))
                lower_raw.append((x, bot_edge))
                std_max_x = x   # track rightmost standard-branch detection

        return upper_raw, lower_raw, std_max_x

    @staticmethod
    def _white_runs(col: np.ndarray, h: int) -> list[tuple[int, int]]:
        """Return list of (start_y, end_y) for runs of 255 in *col*."""
        runs: list[tuple[int, int]] = []
        in_run = False
        start = 0
        for i in range(h):
            if col[i] == 255:
                if not in_run:
                    in_run = True
                    start = i
            else:
                if in_run:
                    in_run = False
                    runs.append((start, i - 1))
        if in_run:
            runs.append((start, h - 1))
        return runs

    # ------------------------------------------------------------------
    # Canny mode
    # ------------------------------------------------------------------

    def _build_edge_map(self, preprocessed: np.ndarray) -> np.ndarray:
        otsu_thresh, _ = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        low  = max(0,   int(0.5 * otsu_thresh))
        high = min(255, int(1.5 * otsu_thresh))
        if high - low < 20:
            low = max(0, high - 40)
        edges = cv2.Canny(preprocessed, low, high)
        return cv2.dilate(edges, self._h_kernel, iterations=self.dilate_iters)

    def _scan_canny(
        self,
        edge_map: np.ndarray,
        preprocessed: np.ndarray,
    ) -> tuple[list, list]:
        h, w = edge_map.shape
        min_score = self.min_arm_gap_contrast * 255.0
        upper_raw: list[tuple[int, int]] = []
        lower_raw: list[tuple[int, int]] = []

        for x in range(w):
            ys = np.where(edge_map[:, x] > 0)[0]
            if len(ys) < 2:
                continue

            col = preprocessed[:, x].astype(float)
            best_score = -np.inf
            best_top: int | None = None
            best_bot: int | None = None

            for i in range(len(ys) - 1):
                for j in range(i + 1, len(ys)):
                    top = int(ys[i])
                    bot = int(ys[j])
                    if bot - top < self.min_gap:
                        continue
                    mid_y   = (top + bot) // 2
                    gap_int = float(col[mid_y])
                    arm_top = float(col[max(0, top - 3)])
                    arm_bot = float(col[min(h - 1, bot + 3)])
                    score = (
                        min(gap_int - arm_top, gap_int - arm_bot)
                        if self.bright_crack
                        else min(arm_top - gap_int, arm_bot - gap_int)
                    )
                    if score > best_score:
                        best_score = score
                        best_top   = top
                        best_bot   = bot

            if best_top is not None and best_score >= min_score:
                upper_raw.append((x, best_top))
                lower_raw.append((x, best_bot))

        return upper_raw, lower_raw

    # ------------------------------------------------------------------
    # Near-tip extension scan
    # ------------------------------------------------------------------

    def _scan_tip_extension(
        self,
        preprocessed: np.ndarray,
        seeds: dict[int, tuple[int, int]],
    ) -> tuple[list, list]:
        """
        Lookahead guided scan for columns just beyond the detected crack tip.

        At the tip the gap may be only 1-2 px and the gap pixels can still
        be above the global Otsu threshold (they scatter light from both
        arms).  Using a threshold boosted by +30 above Otsu makes those
        transitional pixels appear dark so that two distinct binary runs
        become visible.  A relaxed min_run (half of normal, ≥4 px) is
        applied because the seed position from polynomial extrapolation
        already confirms where the arm should be.

        *seeds* maps x → (exp_uy, exp_ly) from polynomial extrapolation.
        """
        otsu_val, _ = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Tighten threshold beyond Otsu so that 1-2 px gap pixels (which are
        # typically 15-25 % below the arm brightness but still above the
        # global Otsu cutoff) are rendered dark.
        thresh = min(220, max(int(otsu_val) + 30, self.min_threshold + 30))
        _, binary = cv2.threshold(preprocessed, thresh, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

        h = binary.shape[0]
        min_run_ext = max(self.min_run // 4, 2)
        _EXT_GUIDE = 35
        upper_ext: list[tuple[int, int]] = []
        lower_ext: list[tuple[int, int]] = []

        for x in sorted(seeds):
            exp_uy, exp_ly = seeds[x]
            col   = binary[:, x]
            col_g = preprocessed[:, x].astype(float)

            # Upper arm
            y0u = max(0, exp_uy - _EXT_GUIDE)
            y1u = min(h - 1, exp_uy + _EXT_GUIDE)
            runs_u = self._white_runs(col[y0u : y1u + 1], y1u - y0u + 1)
            valid_u = [(s + y0u, e + y0u) for s, e in runs_u
                       if (e - s + 1) >= min_run_ext]
            upper_run = None
            for s, e in valid_u:
                if float(np.mean(col_g[s : e + 1])) >= self.min_arm_brightness:
                    upper_run = (s, e)
                    break
            if upper_run is None:
                break  # upper arm not found → stop extending

            # Internal top position: last pixel >= min_arm_brightness.
            # Used for lower-arm window placement and gap termination check
            # (preserves original detection reach near the tip).
            top_inner = upper_run[0]
            for yi in range(upper_run[0], upper_run[1] + 1):
                if col_g[yi] >= self.min_arm_brightness:
                    top_inner = yi

            # Visualisation position: anchor+1 style, consistent with main scan.
            # Placed at the grey interface just below the clearly-white region,
            # independent of Otsu drift.
            _EDGE_T_EXT = 120
            top_vis = upper_run[1]
            for yi in range(upper_run[0], upper_run[1] + 1):
                if col_g[yi] >= _EDGE_T_EXT:
                    top_vis = yi
            top_edge = min(top_vis + 1, h - 1)   # stored in cumul for drawing

            # Lower arm (strictly below top_inner to preserve original window)
            y0l = max(top_inner + 1, exp_ly - _EXT_GUIDE)
            y1l = min(h - 1, exp_ly + _EXT_GUIDE)
            if y0l > y1l:
                break
            runs_l = self._white_runs(col[y0l : y1l + 1], y1l - y0l + 1)
            valid_l = [(s + y0l, e + y0l) for s, e in runs_l
                       if (e - s + 1) >= min_run_ext]
            lower_run = None
            for s, e in valid_l:
                if float(np.mean(col_g[s : e + 1])) >= self.min_lower_brightness:
                    lower_run = (s, e)
                    break
            if lower_run is None:
                break  # lower arm not found → stop extending

            # Internal bot position for gap termination check.
            bot_inner = lower_run[1]
            for yi in range(lower_run[0], lower_run[1] + 1):
                if col_g[yi] >= self.min_lower_brightness:
                    bot_inner = yi
                    break

            # Visualisation position: anchor-1 style.
            bot_vis = lower_run[0]
            for yi in range(lower_run[0], lower_run[1] + 1):
                if col_g[yi] >= _EDGE_T_EXT:
                    bot_vis = yi
                    break
            bot_edge = max(0, bot_vis - 1)   # stored in cumul for drawing

            if (bot_inner - top_inner) < 1:
                break  # arms touching (checked on inner positions) — stop

            upper_ext.append((x, top_edge))
            lower_ext.append((x, bot_edge))

        return upper_ext, lower_ext

    # ------------------------------------------------------------------
    # Gradient-based sub-pixel tip methods
    # ------------------------------------------------------------------

    @staticmethod
    def _gradient_edge(
        col_smooth: np.ndarray,
        center: int,
        window: int,
        direction: str,
    ) -> float | None:
        """
        Find sub-pixel edge position in a smoothed 1-D intensity profile.

        direction='falling' → bright-to-dark (upper arm bottom edge).
        direction='rising'  → dark-to-bright (lower arm top edge).

        Uses parabolic interpolation around the gradient extremum.
        """
        n = len(col_smooth)
        y0 = max(1, center - window)
        y1 = min(n - 2, center + window)
        if y1 - y0 < 3:
            return None
        seg = col_smooth[y0 : y1 + 1]
        grad = np.gradient(seg)
        idx = int(np.argmin(grad) if direction == 'falling' else np.argmax(grad))
        if idx == 0 or idx >= len(grad) - 1:
            return None
        g0, g1, g2 = grad[idx - 1], grad[idx], grad[idx + 1]
        denom = g0 - 2.0 * g1 + g2
        delta = 0.5 * (g0 - g2) / denom if abs(denom) > 1e-6 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        return float(y0 + idx) + delta

    def _scan_tip_gradient(
        self,
        preprocessed: np.ndarray,
        seeds: dict[int, tuple[int, int]],
        min_subpixel_gap: float = 0.3,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], float | None]:
        """
        Gradient-based scan for crack-tip columns beyond the binary detection limit.

        For each seeded column, finds the upper-arm bottom and lower-arm top via
        gradient extrema on the Gaussian-smoothed intensity profile.  Returns
        integer-rounded boundary tuples (compatible with the existing pipeline)
        plus a sub-pixel tip x estimate from gap-weighted interpolation.
        """
        h = preprocessed.shape[0]
        upper_ext: list[tuple[int, int]] = []
        lower_ext: list[tuple[int, int]] = []
        gaps: list[tuple[int, float]] = []   # (x, gap_subpixel)
        _WIN = 18

        for x in sorted(seeds):
            exp_uy, exp_ly = seeds[x]
            col_g = preprocessed[:, x].astype(float)
            col_s = _smooth1d(col_g, sigma=1.2)

            top_sp = self._gradient_edge(col_s, exp_uy, _WIN, 'falling')
            if top_sp is None:
                break
            bot_sp = self._gradient_edge(col_s, exp_ly, _WIN, 'rising')
            if bot_sp is None:
                break

            gap_sp = bot_sp - top_sp
            if gap_sp < min_subpixel_gap:
                # Record negative/zero gap for interpolation, then stop
                gaps.append((x, gap_sp))
                break

            # Arm brightness sanity (relaxed: 82 % of normal threshold)
            y_au0 = max(0, int(top_sp) - 4)
            y_au1 = max(y_au0, int(top_sp))
            if float(np.mean(col_g[y_au0 : y_au1 + 1])) < self.min_arm_brightness * 0.82:
                break
            y_al0 = min(h - 1, int(bot_sp))
            y_al1 = min(h - 1, int(bot_sp) + 4)
            if float(np.mean(col_g[y_al0 : y_al1 + 1])) < self.min_lower_brightness * 0.82:
                break

            upper_ext.append((x, int(round(top_sp))))
            lower_ext.append((x, int(round(bot_sp))))
            gaps.append((x, gap_sp))

        # Sub-pixel tip x: linear interpolation between last positive-gap column
        # and the first zero-gap column.
        subpixel_tip_x: float | None = None
        if gaps:
            last_pos = [(x, g) for x, g in gaps if g >= min_subpixel_gap]
            first_neg = [(x, g) for x, g in gaps if g < min_subpixel_gap]
            if last_pos:
                x_last, g_last = last_pos[-1]
                if first_neg:
                    x_next, g_next = first_neg[0]
                    span = g_last - g_next
                    subpixel_tip_x = (
                        x_last + g_last / span * (x_next - x_last)
                        if span > 1e-6 else float(x_last)
                    )
                else:
                    # Ran out of search range; tip is somewhere beyond last column
                    subpixel_tip_x = float(x_last) + g_last / (g_last + 1.0)

        return upper_ext, lower_ext, subpixel_tip_x

    def _scan_tip_valley(
        self,
        preprocessed: np.ndarray,
        seeds: dict[int, tuple[int, int]],
        min_valley_frac: float = 0.05,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], float | None]:
        """
        Bimodal valley detection — most sensitive crack-tip extension layer.

        Looks for a local intensity minimum (valley) between the two arm peaks
        in each column.  Works even when the gap is sub-pixel and the arms are
        gray rather than bright-white: the valley only needs to be 5% darker
        than the surrounding arm material.

        Algorithm per column
        --------------------
        1. Extract the intensity profile over [exp_uy-WIN, exp_ly+WIN].
        2. Gaussian-smooth to suppress pixel noise.
        3. Find the maximum in the upper half  → upper-arm peak.
        4. Find the maximum in the lower half  → lower-arm peak.
        5. Find the minimum between those two peaks → valley.
        6. If (arm_mean - valley) / arm_mean >= min_valley_frac → gap detected.
        7. Sub-pixel arm boundaries from _gradient_edge.
        """
        h = preprocessed.shape[0]
        upper_ext: list[tuple[int, int]] = []
        lower_ext: list[tuple[int, int]] = []
        gaps: list[tuple[int, float]] = []   # (x, valley_frac)
        _WIN = 14

        for x in sorted(seeds):
            exp_uy, exp_ly = seeds[x]

            # If polynomial predicts arms nearly touching, widen around midpoint
            if exp_ly - exp_uy < 6:
                mid = (exp_uy + exp_ly) // 2
                exp_uy = max(0, mid - 8)
                exp_ly = min(h - 1, mid + 8)

            col_g = preprocessed[:, x].astype(float)
            col_s = _smooth1d(col_g, sigma=1.5)

            y0 = max(0, exp_uy - _WIN)
            y1 = min(h - 1, exp_ly + _WIN)
            if y1 - y0 < 5:
                break
            profile = col_s[y0 : y1 + 1]
            n = len(profile)

            # Split profile at midpoint so upper/lower halves never overlap,
            # even when the polynomial predicts arms nearly touching.
            mid_p = n // 2
            u_end = max(1, mid_p - 1)
            l_start = min(n - 2, mid_p + 1)
            if u_end < 1 or l_start >= n - 1:
                break

            u_idx = int(np.argmax(profile[:u_end + 1]))
            u_val = float(profile[u_idx])
            l_idx = int(np.argmax(profile[l_start:])) + l_start
            l_val = float(profile[l_idx])

            if l_idx <= u_idx:
                break  # peaks in wrong order

            # Both arms individually must be above the background floor.
            # Using 0.75 of threshold (not 0.65) to avoid false bimodal
            # patterns in the background region beyond the specimen.
            arm_min = self.min_arm_brightness * 0.75
            if u_val < arm_min or l_val < arm_min:
                break

            arm_mean = (u_val + l_val) / 2.0

            # Valley between the two peaks
            between = profile[u_idx : l_idx + 1]
            v_rel = int(np.argmin(between))
            v_val = float(between[v_rel])

            # Relative AND absolute depth: at least 5% relative and 7 DN
            # absolute dip — suppresses texture noise in gray-zone.
            valley_frac = (arm_mean - v_val) / arm_mean if arm_mean > 0 else 0.0
            abs_dip = arm_mean - v_val

            if valley_frac < min_valley_frac or abs_dip < 7.0:
                gaps.append((x, valley_frac))
                break  # arms touching — true tip

            # Sub-pixel arm boundary positions via gradient edge
            top_sp = self._gradient_edge(col_s, exp_uy, _WIN, 'falling')
            if top_sp is None:
                top_sp = float(y0 + u_idx)
            bot_sp = self._gradient_edge(col_s, exp_ly, _WIN, 'rising')
            if bot_sp is None:
                bot_sp = float(y0 + l_idx)

            if top_sp >= bot_sp:
                v_abs = y0 + u_idx + v_rel
                top_sp = float(v_abs) - 0.5
                bot_sp = float(v_abs) + 0.5

            upper_ext.append((x, int(round(top_sp))))
            lower_ext.append((x, int(round(bot_sp))))
            gaps.append((x, valley_frac))

        # Sub-pixel tip x: interpolate where valley_frac → 0
        subpixel_tip_x: float | None = None
        pos = [(x, vf) for x, vf in gaps if vf >= min_valley_frac]
        neg = [(x, vf) for x, vf in gaps if vf < min_valley_frac]
        if pos:
            x_last, vf_last = pos[-1]
            if neg:
                x_next, vf_next = neg[0]
                span = vf_last - vf_next
                subpixel_tip_x = (
                    float(x_last) + vf_last / span * (x_next - x_last)
                    if span > 1e-6 else float(x_last)
                )
            else:
                subpixel_tip_x = float(x_last) + vf_last / (vf_last + min_valley_frac) * 0.5

        return upper_ext, lower_ext, subpixel_tip_x

    # ------------------------------------------------------------------
    # Dark-pixel tip scan (4th extension layer)
    # ------------------------------------------------------------------

    def _scan_tip_dark(
        self,
        preprocessed: np.ndarray,
        seeds: dict[int, tuple[int, int]],
        dark_frac: float = 0.90,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], float | None]:
        """
        En hassas çatlak ucu uzantısı.

        Dar uç bölgesinde CLAHE lokal normalizasyonu kol parlaklığını global
        min_arm_brightness eşiğinin altına (örn. 145→112 DN) düşürebilir.
        Bu yüzden global eşik değil, sütunun kendi maksimum parlaklığı referans
        alınır.  Boşluk pikseli sütun maksimumunun %90 altındaysa → çatlak.

        Polinom inversiyonu (exp_uy >= exp_ly) durumunda midpoint penceresi.
        """
        h = preprocessed.shape[0]
        upper_ext: list[tuple[int, int]] = []
        lower_ext: list[tuple[int, int]] = []
        last_x: int | None = None
        _SEARCH = 8  # kenar arama penceresi

        for x in sorted(seeds):
            exp_uy, exp_ly = seeds[x]
            col_g = preprocessed[:, x].astype(float)

            # Sütunun kendi en yüksek parlaklığı = kol referansı.
            # Global min_arm_brightness YERINE col_max kullanıyoruz.
            col_max = float(np.max(col_g))
            if col_max < 55:
                break  # tamamen karanlık sütun → numune dışı → dur

            threshold = col_max * dark_frac  # örn. 112 * 0.90 = 100.8

            # Polinom inversiyonu varsa midpoint penceresi
            if exp_uy >= exp_ly:
                mid = (exp_uy + exp_ly) // 2
                exp_uy = max(0, mid - 4)
                exp_ly = min(h - 1, mid + 4)

            # Boşluk bölgesini kontrol et
            g0 = max(0, exp_uy)
            g1 = min(h - 1, exp_ly)
            if g0 > g1:
                g0, g1 = g1, g0
            if g0 == g1:
                g0 = max(0, g0 - 2)
                g1 = min(h - 1, g1 + 2)

            gap_col = col_g[g0:g1 + 1]
            if len(gap_col) == 0:
                break

            n_check = max(1, min(2, len(gap_col)))
            gap_val = float(np.mean(np.sort(gap_col)[:n_check]))

            if gap_val < threshold:
                # Kol kenarlarını bul: gap'in üstünde son parlak piksel,
                # altında ilk parlak piksel.
                arm_thresh = col_max * 0.78
                top_arm = g0   # arm/gap interface (internal, for gap check)
                for yi in range(g0, max(-1, g0 - _SEARCH - 1), -1):
                    if col_g[yi] >= arm_thresh:
                        top_arm = yi
                        break

                bot_arm = g1   # arm/gap interface (internal, for gap check)
                for yi in range(g1, min(h, g1 + _SEARCH + 1)):
                    if col_g[yi] >= arm_thresh:
                        bot_arm = yi
                        break

                if top_arm >= bot_arm:
                    bot_arm = top_arm + 1

                # Visualisation: report grey interface pixel (one step into
                # the gap from each arm) — consistent with binary scan anchor+1/-1.
                top_edge = min(top_arm + 1, h - 1)
                bot_edge = max(0, bot_arm - 1)
                if top_edge >= bot_edge:
                    continue   # gap too small to place markers → skip column

                upper_ext.append((x, top_edge))
                lower_ext.append((x, bot_edge))
                last_x = x
            else:
                break  # kollar burada birleşiyor → dur

        sp_x = float(last_x) if last_x is not None else None
        return upper_ext, lower_ext, sp_x

    # ------------------------------------------------------------------
    # Shared: polynomial boundary fitting
    # ------------------------------------------------------------------

    def _fit_boundary(
        self, pts: np.ndarray, x_min_extend: int | None = None
    ) -> np.ndarray | None:
        if pts is None or len(pts) < self.poly_degree + 1:
            return None

        xs, ys = pts[:, 0], pts[:, 1]

        q1, q3 = np.percentile(ys, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            mask = (ys >= q1 - 1.5 * iqr) & (ys <= q3 + 1.5 * iqr)
            xs, ys = xs[mask], ys[mask]

        if len(xs) < self.poly_degree + 1:
            return None

        try:
            coeffs = np.polyfit(xs, ys, self.poly_degree)
        except (np.linalg.LinAlgError, ValueError):
            return None

        x_start = int(xs.min())
        if x_min_extend is not None and x_min_extend < x_start:
            x_start = x_min_extend
        x_range = np.arange(x_start, int(xs.max()) + 1)
        y_fitted = np.clip(np.polyval(coeffs, x_range), 0, 1e6).astype(int)
        return np.column_stack([x_range, y_fitted])
