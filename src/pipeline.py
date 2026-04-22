"""High-level tracking pipeline -- the single entry point used by both the
Streamlit dashboard and the CLI ``demo.py``.

The pipeline owns:

* one :class:`YOLODetector`
* one :class:`ByteTracker`
* four analytics components (counter, heatmap, trajectory, speed)
* a per-frame composition step that produces the annotated output frame.

Keeping this assembly in a single class avoids duplicating wiring between
the UI and the CLI and makes it trivial to drop the whole pipeline into
another application (e.g. a FastAPI endpoint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .analytics import MotionHeatmap, ObjectCounter, SpeedEstimator, TrajectoryStore
from .detector import YOLODetector
from .tracker import ByteTracker
from .visualization import draw_tracks, draw_trajectories


@dataclass
class PipelineConfig:
    weights: str = "yolov8n.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    imgsz: int = 640
    device: Optional[str] = None
    class_filter: Optional[Sequence[str]] = None

    # Tracker
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    match_thresh: float = 0.8
    max_lost_frames: int = 30

    # Analytics toggles
    show_boxes: bool = True
    show_trails: bool = True
    show_heatmap_overlay: bool = False
    heatmap_alpha: float = 0.5
    trail_length: int = 60


@dataclass
class FrameResult:
    annotated: np.ndarray
    heatmap_only: np.ndarray
    tracks: list = field(default_factory=list)
    counts_current: dict = field(default_factory=dict)
    counts_total: dict = field(default_factory=dict)
    speed_summary: dict = field(default_factory=dict)


class TrackingPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None, fps: float = 30.0) -> None:
        self.config = config or PipelineConfig()
        self.fps = fps
        self._init_components()

    def _init_components(self) -> None:
        c = self.config
        self.detector = YOLODetector(
            weights=c.weights,
            device=c.device,
            conf_threshold=c.conf_threshold,
            iou_threshold=c.iou_threshold,
            class_filter=c.class_filter,
            imgsz=c.imgsz,
        )
        self.tracker = ByteTracker(
            high_thresh=c.high_thresh,
            low_thresh=c.low_thresh,
            match_thresh=c.match_thresh,
            max_lost_frames=c.max_lost_frames,
        )
        self.counter = ObjectCounter()
        self.trajectory = TrajectoryStore(max_points=c.trail_length)
        self.speed = SpeedEstimator(fps=self.fps)
        self._heatmap: Optional[MotionHeatmap] = None

    def reset(self) -> None:
        """Reset all tracking state -- used when a new video is uploaded."""
        self._init_components()

    # ------------------------------------------------------------------
    # Per-frame step
    # ------------------------------------------------------------------
    def step(self, frame: np.ndarray) -> FrameResult:
        if self._heatmap is None or self._heatmap._accum.shape != frame.shape[:2]:
            self._heatmap = MotionHeatmap(frame.shape[:2])

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)

        self.counter.update(tracks)
        self.trajectory.update(tracks)
        self.speed.update(tracks)
        self._heatmap.update(tracks)

        annotated = frame.copy()
        if self.config.show_trails:
            annotated = draw_trajectories(annotated, self.trajectory.trails())
        if self.config.show_boxes:
            annotated = draw_tracks(annotated, tracks, speed_estimator=self.speed)
        if self.config.show_heatmap_overlay:
            annotated = self._heatmap.render(annotated, alpha=self.config.heatmap_alpha)

        heatmap_only = self._heatmap.render(None)

        return FrameResult(
            annotated=annotated,
            heatmap_only=heatmap_only,
            tracks=tracks,
            counts_current=self.counter.current,
            counts_total=self.counter.total_unique,
            speed_summary=self.speed.summary(),
        )

    # ------------------------------------------------------------------
    # Whole-video convenience
    # ------------------------------------------------------------------
    def process_video(
        self,
        source: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        progress=None,
    ) -> dict:
        """Run the pipeline on a full video file.

        ``progress`` is an optional callable ``progress(frame_idx, total, result)``
        used by the Streamlit UI to stream frames into the page.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {source}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps = fps
        self.speed.fps = fps
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        last_result: Optional[FrameResult] = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break
                result = self.step(frame)
                last_result = result
                if writer is not None:
                    writer.write(result.annotated)
                if progress is not None:
                    progress(frame_idx, total, result)
                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        return {
            "frames_processed": frame_idx,
            "final": last_result,
        }
