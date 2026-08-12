"""
video_reader.py
---------------
Reads video files frame by frame and exposes metadata.
Always returns grayscale frames for consistent downstream processing.
"""

import cv2
import numpy as np
from pathlib import Path


class VideoReader:
    """Opens a video file and yields grayscale frames with metadata."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame as a grayscale numpy array."""
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, self._to_gray(frame)

    def get_frame(self, index: int) -> np.ndarray | None:
        """Seek to and return the frame at the given index."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        return self._to_gray(frame) if ret else None

    def get_first_frame(self) -> np.ndarray:
        """Return the very first frame without changing the read position."""
        pos = self.current_index
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frame = self._cap.read()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        return self._to_gray(frame)

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def reset(self):
        """Rewind to the beginning."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def seek(self, frame_index: int):
        """Seek to an arbitrary frame index."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))

    @property
    def current_index(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def release(self):
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self):
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoReader('{self.video_path.name}', "
            f"{self.width}x{self.height}, {self.fps:.1f} fps, "
            f"{self.frame_count} frames)"
        )
