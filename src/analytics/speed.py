"""Per-track pixel-space speed and direction estimation.

Speed is reported in *pixels per second*. If the caller knows a real-world
scale (e.g. from a calibration checkerboard), they can multiply by
``meters_per_pixel`` to get m/s.

Direction is a 2-D unit vector (dx, dy) averaged over the last ``window``
samples, which prevents the arrow from flickering on noisy detections.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Tuple

import numpy as np


class SpeedEstimator:
    def __init__(self, fps: float = 30.0, window: int = 8) -> None:
        self.fps = max(1.0, float(fps))
        self.window = window
        self._centers: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._speed: Dict[int, float] = {}
        self._direction: Dict[int, Tuple[float, float]] = {}

    def update(self, tracks: Iterable) -> None:
        for t in tracks:
            c = t.center()
            self._centers[t.track_id].append(c)
            self._compute(t.track_id)

    def _compute(self, track_id: int) -> None:
        pts = self._centers[track_id]
        if len(pts) < 2:
            self._speed[track_id] = 0.0
            self._direction[track_id] = (0.0, 0.0)
            return
        arr = np.asarray(pts, dtype=np.float32)
        deltas = np.diff(arr, axis=0)
        # pixels / frame averaged over window, then convert to pixels / second
        mean_delta = deltas.mean(axis=0)
        speed_px_per_frame = float(np.linalg.norm(mean_delta))
        self._speed[track_id] = speed_px_per_frame * self.fps

        n = np.linalg.norm(mean_delta)
        if n > 1e-3:
            self._direction[track_id] = (
                float(mean_delta[0] / n),
                float(mean_delta[1] / n),
            )
        else:
            self._direction[track_id] = (0.0, 0.0)

    def speed_of(self, track_id: int) -> float:
        return self._speed.get(track_id, 0.0)

    def direction_of(self, track_id: int) -> Tuple[float, float]:
        return self._direction.get(track_id, (0.0, 0.0))

    def summary(self) -> Dict[str, float]:
        """Aggregate metrics for the analytics panel."""
        if not self._speed:
            return {"mean_speed": 0.0, "max_speed": 0.0, "num_moving": 0}
        speeds = np.array(list(self._speed.values()), dtype=np.float32)
        return {
            "mean_speed": float(speeds.mean()),
            "max_speed": float(speeds.max()),
            "num_moving": int((speeds > 10.0).sum()),  # >10 px/s considered moving
        }
