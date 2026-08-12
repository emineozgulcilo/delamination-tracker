"""
export.py
---------
Saves all outputs produced by a processing run:

  1. annotated_video.mp4  — original frames with overlays composited
  2. measurements.csv     — per-frame numeric data
  3. length_vs_time.png   — matplotlib figure of crack length over time
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend; must come before pyplot import
import matplotlib.pyplot as plt

try:
    from measurement import Measurement
except ModuleNotFoundError:
    from tracker.measurement import Measurement


class Exporter:
    """
    Writes all analysis outputs to *output_dir*.

    Parameters
    ----------
    output_dir : str | Path
        Directory that will be created if it does not exist.
    """

    def __init__(self, output_dir: str | Path = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Video writer
    # ------------------------------------------------------------------

    def create_video_writer(
        self,
        filename: str,
        fps: float,
        width: int,
        height: int,
    ) -> cv2.VideoWriter:
        """Return an open VideoWriter for the annotated output video."""
        path = self.output_dir / filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise IOError(f"Cannot create video writer: {path}")
        print(f"[Export] Writing annotated video -> {path}")
        return writer

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def export_csv(
        self,
        records: List[Measurement],
        filename: str = "measurements.csv",
    ) -> Path:
        """Write all measurement records to a CSV file."""
        path = self.output_dir / filename
        fieldnames = [
            "Frame",
            "Time(s)",
            "CrackLength(mm)",
            "CrackLength(px)",
            "CrackTipX",
            "CrackTipY",
            "GapAtTip(px)",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for m in records:
                writer.writerow({
                    "Frame":             m.frame,
                    "Time(s)":           m.time_s,
                    "CrackLength(mm)":   m.crack_length_mm,
                    "CrackLength(px)":   m.crack_length_px,
                    "CrackTipX":         m.crack_tip_x,
                    "CrackTipY":         m.crack_tip_y,
                    "GapAtTip(px)":      m.gap_at_tip_px,
                })
        print(f"[Export] CSV -> {path}  ({len(records)} rows)")
        return path

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def export_plot(
        self,
        records: List[Measurement],
        filename: str = "length_vs_time.png",
    ) -> Path:
        """Generate a Crack Length vs Time plot and save it as PNG."""
        path = self.output_dir / filename

        times   = [m.time_s for m in records]
        lengths = [m.crack_length_mm for m in records]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, lengths, color="#1f77b4", linewidth=1.5,
                label="Crack length (mm)")

        # Mark the point of maximum propagation
        if lengths:
            max_len = max(lengths)
            max_t   = times[lengths.index(max_len)]
            ax.axhline(max_len, color="red", linestyle="--", linewidth=0.8,
                       label=f"Max = {max_len:.2f} mm")
            ax.plot(max_t, max_len, "ro", markersize=6)

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Crack Length (mm)", fontsize=12)
        ax.set_title("Delamination Crack Propagation — Length vs Time",
                     fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

        print(f"[Export] Plot -> {path}")
        return path

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self, records: List[Measurement], fps: float):
        """Print a short statistics summary to stdout."""
        if not records:
            print("[Export] No measurements to summarise.")
            return

        lengths = [m.crack_length_mm for m in records]
        times   = [m.time_s for m in records]
        print("\n" + "=" * 50)
        print("  DELAMINATION TRACKING SUMMARY")
        print("=" * 50)
        print(f"  Frames processed : {len(records)}")
        print(f"  Duration         : {times[-1]:.3f} s")
        print(f"  Initial length   : {lengths[0]:.3f} mm")
        print(f"  Final length     : {lengths[-1]:.3f} mm")
        print(f"  Maximum length   : {max(lengths):.3f} mm")
        if times[-1] > 0:
            avg_rate = (lengths[-1] - lengths[0]) / times[-1]
            print(f"  Avg crack rate   : {avg_rate:.3f} mm/s")
        print("=" * 50)
