"""Unit tests for the tracker + analytics pipeline.

These tests exercise the deterministic, numpy-only layers (Kalman filter,
IoU matching, ByteTracker, analytics). They never touch YOLOv8 weights so
they run in under a second and are safe in CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.analytics import MotionHeatmap, ObjectCounter, SpeedEstimator, TrajectoryStore
from src.tracker import ByteTracker
from src.tracker.matching import iou, linear_assignment


@dataclass
class _FakeDetection:
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str

    @property
    def tlwh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def _make_det(x, y, w=40, h=80, conf=0.9, cls_id=0, cls_name="person") -> _FakeDetection:
    return _FakeDetection(
        bbox=np.array([x, y, x + w, y + h], dtype=np.float32),
        confidence=conf,
        class_id=cls_id,
        class_name=cls_name,
    )


# ---------------------------------------------------------------------------
# IoU + assignment
# ---------------------------------------------------------------------------
def test_iou_identity_is_one() -> None:
    b = np.array([0, 0, 10, 10])
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_is_zero() -> None:
    a = np.array([0, 0, 10, 10])
    b = np.array([100, 100, 110, 110])
    assert iou(a, b) == 0.0


def test_linear_assignment_handles_empty() -> None:
    cost = np.zeros((0, 0), dtype=np.float32)
    matches, ua, ub = linear_assignment(cost, 0.5)
    assert matches == [] and ua == [] and ub == []


def test_linear_assignment_respects_threshold() -> None:
    # Two tracks, two detections. Distance matrix forces one good match
    # and one bad one that should be rejected by the threshold.
    cost = np.array([[0.1, 0.9], [0.95, 0.2]], dtype=np.float32)
    matches, ua, ub = linear_assignment(cost, 0.3)
    assert set(matches) == {(0, 0), (1, 1)}
    assert ua == [] and ub == []


# ---------------------------------------------------------------------------
# ByteTracker end-to-end
# ---------------------------------------------------------------------------
def test_tracker_maintains_id_across_frames() -> None:
    tracker = ByteTracker(match_thresh=0.8, max_lost_frames=10)
    ids_per_frame = []
    for step in range(10):
        # One object moving diagonally 5 px per frame.
        det = _make_det(100 + step * 5, 100 + step * 5)
        tracks = tracker.update([det])
        assert len(tracks) == 1
        ids_per_frame.append(tracks[0].track_id)

    assert len(set(ids_per_frame)) == 1, "Track id must persist across frames"


def test_tracker_assigns_new_ids_for_new_objects() -> None:
    tracker = ByteTracker()
    tracks_1 = tracker.update([_make_det(10, 10)])
    tracks_2 = tracker.update([_make_det(10, 10), _make_det(300, 300)])
    id1 = tracks_1[0].track_id
    new_ids = [t.track_id for t in tracks_2 if t.track_id != id1]
    assert len(new_ids) == 1


def test_tracker_ages_out_lost_tracks() -> None:
    tracker = ByteTracker(max_lost_frames=3)
    tracker.update([_make_det(10, 10)])
    for _ in range(10):
        tracker.update([])
    # Track should have been aged out -> not in tracked *or* lost anymore.
    assert len(tracker.lost_tracks) == 0


def test_tracker_is_class_aware() -> None:
    """A car detection at the same location as a previous person should get
    a new id, not steal the person's id."""
    tracker = ByteTracker()
    person_tracks = tracker.update([_make_det(100, 100, cls_id=0, cls_name="person")])
    # New frame: a *car* detection overlapping the person.
    car_tracks = tracker.update([_make_det(100, 100, cls_id=2, cls_name="car")])
    assert person_tracks[0].track_id != car_tracks[0].track_id


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def test_counter_counts_unique_ids() -> None:
    counter = ObjectCounter()

    class _T:
        def __init__(self, tid, name):
            self.track_id, self.class_name = tid, name

    counter.update([_T(1, "person"), _T(2, "person"), _T(3, "car")])
    counter.update([_T(1, "person"), _T(4, "person")])
    assert counter.total_unique == {"person": 3, "car": 1}
    assert counter.current == {"person": 2}


def test_heatmap_accumulates_activity() -> None:
    heat = MotionHeatmap(frame_shape=(100, 200))
    assert heat._accum.sum() == 0.0

    class _T:
        def center(self):
            return (50.0, 50.0)

    heat.update([_T()])
    assert heat._accum.sum() > 0


def test_speed_estimator_returns_positive_for_moving_track() -> None:
    estimator = SpeedEstimator(fps=30.0)

    class _T:
        def __init__(self, x):
            self.track_id = 1
            self._x = x

        def center(self):
            return (self._x, 0.0)

    for x in range(0, 100, 10):
        estimator.update([_T(x)])

    speed = estimator.speed_of(1)
    assert speed > 0


def test_trajectory_store_keeps_recent_points_only() -> None:
    store = TrajectoryStore(max_points=5)

    class _T:
        def __init__(self, x):
            self.track_id = 1
            self.class_name = "person"
            self._x = x

        def center(self):
            return (self._x, 0.0)

    for x in range(20):
        store.update([_T(x)])

    assert len(store.trails()[1]) == 5
