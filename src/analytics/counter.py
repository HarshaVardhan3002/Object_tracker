"""Track counting + class breakdown.

The counter tracks two things:

* ``current_counts`` -- how many tracks of each class are visible *right now*.
* ``total_counts``   -- cumulative unique track IDs seen per class over the
  whole session. A track id is only counted once, even if it later
  disappears and is rediscovered as a new id.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, Tuple


class ObjectCounter:
    def __init__(self) -> None:
        self._seen_ids: Dict[str, set[int]] = defaultdict(set)
        self._current: Counter[str] = Counter()

    def update(self, tracks: Iterable) -> None:
        current: Counter[str] = Counter()
        for t in tracks:
            current[t.class_name] += 1
            self._seen_ids[t.class_name].add(t.track_id)
        self._current = current

    @property
    def current(self) -> Dict[str, int]:
        return dict(self._current)

    @property
    def total_unique(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self._seen_ids.items()}

    def as_table(self) -> Tuple[list[str], list[int], list[int]]:
        """Return (classes, current_counts, total_counts) sorted by total."""
        classes = sorted(
            set(self._current) | set(self._seen_ids),
            key=lambda c: -len(self._seen_ids.get(c, set())),
        )
        current = [self._current.get(c, 0) for c in classes]
        total = [len(self._seen_ids.get(c, set())) for c in classes]
        return classes, current, total
