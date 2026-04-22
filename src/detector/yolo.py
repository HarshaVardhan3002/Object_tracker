"""YOLOv8 detector wrapper.

A thin, opinionated layer on top of Ultralytics YOLOv8 that returns a clean
list of :class:`Detection` objects per frame. The wrapper handles:

* Automatic model download on first run (yolov8n/s/m/l/x.pt).
* Confidence + IoU NMS thresholding.
* Optional class filtering (e.g. only "person" and "car").
* Device auto-selection (CUDA if available, else CPU).

The detector is deliberately decoupled from the tracker so either component
can be swapped independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


# COCO class names that YOLOv8 is pretrained on. Kept here so callers can
# reference classes by name without importing ultralytics at module load time.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass
class Detection:
    """A single detection in a frame.

    Attributes:
        bbox: xyxy bounding box as [x1, y1, x2, y2] in pixel coordinates.
        confidence: detector confidence score in [0, 1].
        class_id: integer class id (COCO indexing).
        class_name: human-readable class name.
    """

    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str

    @property
    def tlwh(self) -> np.ndarray:
        """Top-left + width/height format used by the Kalman filter."""
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class YOLODetector:
    """Wraps an Ultralytics YOLOv8 model and returns :class:`Detection` lists."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        class_filter: Optional[Sequence[str]] = None,
        imgsz: int = 640,
    ) -> None:
        # Lazy import so the package imports cleanly without torch installed.
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(weights)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.class_filter = (
            {COCO_CLASSES.index(c) for c in class_filter if c in COCO_CLASSES}
            if class_filter
            else None
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run one forward pass and return detections above the threshold.

        Args:
            frame: BGR image as produced by cv2.VideoCapture.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_id in zip(boxes, confs, clses):
            if self.class_filter is not None and cls_id not in self.class_filter:
                continue
            name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else str(cls_id)
            detections.append(
                Detection(
                    bbox=bbox.astype(np.float32),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=name,
                )
            )
        return detections

    def set_class_filter(self, classes: Optional[Iterable[str]]) -> None:
        """Update the class filter at runtime (used by the Streamlit UI)."""
        if classes is None:
            self.class_filter = None
        else:
            self.class_filter = {
                COCO_CLASSES.index(c) for c in classes if c in COCO_CLASSES
            }
