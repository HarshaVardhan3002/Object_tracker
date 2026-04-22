"""Per-track trajectory storage (ring buffer of recent centers)."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable


class TrajectoryStore:
    def __init__(self, max_points: int = 60) -> None:
        self.max_points = max_points
        self._trails: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.max_points)
        )
        self._classes: dict[int, str] = {}

    def update(self, tracks: Iterable) -> None:
        active_ids = set()
        for t in tracks:
            active_ids.add(t.track_id)
            self._trails[t.track_id].append(t.center())
            self._classes[t.track_id] = t.class_name

        # Shrink trails for tracks we haven't seen in the last update so the
        # memory footprint stays bounded on long videos.
        stale = [tid for tid in self._trails if tid not in active_ids and len(self._trails[tid]) > 5]
        for tid in stale:
            # keep a short tail so the trail still fades gracefully
            trail = self._trails[tid]
            while len(trail) > 5:
                trail.popleft()

    def trails(self) -> dict[int, deque[tuple[float, float]]]:
        return self._trails

    def class_of(self, track_id: int) -> str:
        return self._classes.get(track_id, "object")
