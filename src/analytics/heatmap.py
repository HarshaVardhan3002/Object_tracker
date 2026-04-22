"""Accumulated motion heatmap.

Each frame, we stamp a Gaussian blob at the center of every tracked object
into a running 2-D accumulator. The accumulator decays slowly so the heatmap
adapts to changing scene activity rather than saturating forever.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


class MotionHeatmap:
    def __init__(
        self,
        frame_shape: tuple[int, int],
        blob_sigma: float = 25.0,
        decay: float = 0.995,
    ) -> None:
        """``frame_shape`` is (height, width) in pixels."""
        self.height, self.width = frame_shape
        self.blob_sigma = blob_sigma
        self.decay = decay
        self._accum = np.zeros((self.height, self.width), dtype=np.float32)

    def reset(self) -> None:
        self._accum.fill(0.0)

    def update(self, tracks: Iterable) -> None:
        # Decay the whole map slightly each frame so old activity fades.
        self._accum *= self.decay
        if self._accum.max() > 1e5:  # safety -- avoid runaway
            self._accum *= 0.5

        for t in tracks:
            cx, cy = t.center()
            x, y = int(cx), int(cy)
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
            # Fast approximation: paint a small hot square then blur later.
            radius = max(2, int(self.blob_sigma / 5))
            cv2.circle(self._accum, (x, y), radius, 1.0, -1)

    def render(self, frame: np.ndarray | None = None, alpha: float = 0.5) -> np.ndarray:
        """Return the heatmap as a colored image.

        If ``frame`` is provided, the heatmap is overlaid onto it with the
        given alpha (useful for the Streamlit UI).
        """
        blurred = cv2.GaussianBlur(self._accum, (0, 0), self.blob_sigma)
        norm = blurred / (blurred.max() + 1e-6)
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        if frame is None:
            return color

        # Resize the heat map to match the frame in case the user changed size.
        if color.shape[:2] != frame.shape[:2]:
            color = cv2.resize(color, (frame.shape[1], frame.shape[0]))
        return cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0)
