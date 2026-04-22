"""ByteTrack-inspired multi-object tracker.

This is a from-scratch, readable implementation of the *two-stage association*
idea from `ByteTrack <https://arxiv.org/abs/2110.06864>`_:

1. Predict all active tracks one step forward with the Kalman filter.
2. First association: match tracks with *high-confidence* detections via
   IoU + Hungarian assignment.
3. Second association: match the still-unmatched tracks with the *low-confidence*
   detections that would otherwise be thrown away. This is what lets ByteTrack
   keep hold of partially occluded objects where the detector confidence dips.
4. Unmatched high-confidence detections start new tracks.
5. Unmatched tracks age out after ``max_lost_frames``.

The tracker is class-aware: tracks are only matched to detections of the same
class. This prevents a "car" ID from being reassigned to an overlapping
"person" box -- a small but meaningful quality win over vanilla SORT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .kalman_filter import KalmanFilter
from .matching import iou_distance, linear_assignment


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class Track:
    """A single tracked object."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    tlwh: np.ndarray  # top-left x, top-left y, width, height
    kf: KalmanFilter
    mean: np.ndarray = field(default=None)  # type: ignore
    covariance: np.ndarray = field(default=None)  # type: ignore
    state: TrackState = TrackState.NEW
    frame_id: int = 0
    start_frame: int = 0
    time_since_update: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        x, y, w, h = tlwh
        return np.array([x + w / 2.0, y + h / 2.0, w / h, h], dtype=np.float32)

    @staticmethod
    def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
        cx, cy, a, h = xyah
        w = a * h
        return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float32)

    def xyxy(self) -> np.ndarray:
        x, y, w, h = self.tlwh
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    def center(self) -> tuple[float, float]:
        x, y, w, h = self.tlwh
        return (x + w / 2.0, y + h / 2.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self, frame_id: int) -> None:
        self.mean, self.covariance = self.kf.initiate(self.tlwh_to_xyah(self.tlwh))
        self.state = TrackState.TRACKED
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0
        self.history.append(self.center())

    def predict(self) -> None:
        if self.state != TrackState.REMOVED:
            self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
            self.tlwh = self.xyah_to_tlwh(self.mean[:4])

    def update(self, detection_tlwh: np.ndarray, confidence: float, frame_id: int) -> None:
        self.mean, self.covariance = self.kf.update(
            self.mean,
            self.covariance,
            self.tlwh_to_xyah(detection_tlwh),
        )
        self.tlwh = self.xyah_to_tlwh(self.mean[:4])
        self.state = TrackState.TRACKED
        self.confidence = confidence
        self.frame_id = frame_id
        self.time_since_update = 0
        self.history.append(self.center())

    def mark_lost(self) -> None:
        self.state = TrackState.LOST
        self.time_since_update += 1


class ByteTracker:
    """Two-stage IoU-based tracker (class-aware)."""

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_lost_frames: int = 30,
        min_box_area: float = 10.0,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames
        self.min_box_area = min_box_area

        self._tracked: list[Track] = []
        self._lost: list[Track] = []
        self._next_id = 1
        self._frame_id = 0
        self._kf = KalmanFilter()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def update(self, detections) -> list[Track]:
        """Update the tracker with detections from one frame.

        ``detections`` is an iterable of :class:`~src.detector.yolo.Detection`.
        Returns the list of currently-tracked tracks (state == TRACKED).
        """
        self._frame_id += 1

        # ---- Split detections by confidence, and filter tiny boxes ----------
        high_dets, low_dets = [], []
        for d in detections:
            w = d.bbox[2] - d.bbox[0]
            h = d.bbox[3] - d.bbox[1]
            if w * h < self.min_box_area:
                continue
            if d.confidence >= self.high_thresh:
                high_dets.append(d)
            elif d.confidence >= self.low_thresh:
                low_dets.append(d)

        # ---- Predict ---------------------------------------------------------
        for t in self._tracked + self._lost:
            t.predict()

        # ---- First association: tracked + lost tracks vs high dets ----------
        pool = self._tracked + self._lost
        matches, u_track, u_det = self._associate(pool, high_dets, self.match_thresh)

        activated: list[Track] = []
        refound: list[Track] = []
        for ti, di in matches:
            t, d = pool[ti], high_dets[di]
            t.update(d.tlwh, d.confidence, self._frame_id)
            t.class_id, t.class_name = d.class_id, d.class_name
            if t in self._lost:
                refound.append(t)
            else:
                activated.append(t)

        # ---- Second association: remaining tracked tracks vs low dets -------
        remaining_tracks = [pool[i] for i in u_track if pool[i] in self._tracked]
        matches2, u_track2, _ = self._associate(remaining_tracks, low_dets, 0.5)
        for ti, di in matches2:
            t, d = remaining_tracks[ti], low_dets[di]
            t.update(d.tlwh, d.confidence, self._frame_id)
            t.class_id, t.class_name = d.class_id, d.class_name
            activated.append(t)

        # ---- Mark tracks with no match as lost ------------------------------
        for t in remaining_tracks:
            if t.state == TrackState.TRACKED and t.frame_id < self._frame_id:
                t.mark_lost()

        # ---- Spawn new tracks from unmatched high-conf dets -----------------
        new_tracks: list[Track] = []
        unmatched_high = [high_dets[i] for i in u_det]
        for d in unmatched_high:
            t = Track(
                track_id=self._next_id,
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                tlwh=d.tlwh,
                kf=self._kf,
            )
            self._next_id += 1
            t.activate(self._frame_id)
            new_tracks.append(t)

        # ---- Age out lost tracks --------------------------------------------
        for t in list(self._lost):
            if self._frame_id - t.frame_id > self.max_lost_frames:
                t.state = TrackState.REMOVED
                self._lost.remove(t)

        # ---- Sync the two pools ---------------------------------------------
        self._tracked = [t for t in self._tracked if t.state == TrackState.TRACKED]
        self._tracked.extend(activated)
        self._tracked.extend(new_tracks)
        self._tracked.extend(refound)
        for t in refound:
            if t in self._lost:
                self._lost.remove(t)

        newly_lost = [t for t in self._tracked if t.state == TrackState.LOST]
        self._tracked = [t for t in self._tracked if t.state == TrackState.TRACKED]
        self._lost.extend(newly_lost)

        # Deduplicate by id (a track can live in both pools briefly).
        seen: dict[int, Track] = {}
        for t in self._tracked:
            seen[t.track_id] = t
        self._tracked = list(seen.values())

        return self._tracked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _associate(self, tracks: list[Track], detections, thresh: float):
        """Class-aware IoU association."""
        if not tracks or not detections:
            return (
                [],
                list(range(len(tracks))),
                list(range(len(detections))),
            )

        # Build the IoU cost matrix and mask out cross-class pairs.
        track_boxes = [t.xyxy() for t in tracks]
        det_boxes = [d.bbox for d in detections]
        cost = iou_distance(track_boxes, det_boxes)

        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                if t.class_id != d.class_id:
                    cost[i, j] = 1.0

        return linear_assignment(cost, thresh)

    # Expose the current lost pool for callers that want to visualize it.
    @property
    def lost_tracks(self) -> list[Track]:
        return list(self._lost)
