"""Drawing helpers -- bounding boxes, labels, trails, direction arrows."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


# A hand-picked high-contrast palette (RGB). We pick per-id deterministically
# so the same track ID keeps the same color across the whole video.
_PALETTE = np.array(
    [
        [66, 135, 245],
        [245, 66, 138],
        [66, 245, 135],
        [245, 203, 66],
        [173, 66, 245],
        [66, 224, 245],
        [245, 126, 66],
        [87, 245, 66],
        [245, 66, 212],
        [66, 245, 197],
        [245, 173, 66],
        [150, 66, 245],
    ],
    dtype=np.uint8,
)


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministic BGR color for a given track id."""
    rgb = _PALETTE[track_id % len(_PALETTE)]
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable,
    speed_estimator=None,
    show_speed: bool = True,
    show_direction: bool = True,
) -> np.ndarray:
    """Draw bounding boxes, IDs, class labels, and (optionally) speed arrows."""
    out = frame.copy()
    for t in tracks:
        color = color_for_id(t.track_id)
        x1, y1, x2, y2 = map(int, t.xyxy())
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label: "ID-class conf%"
        label = f"#{t.track_id} {t.class_name} {int(t.confidence * 100)}%"
        if show_speed and speed_estimator is not None:
            sp = speed_estimator.speed_of(t.track_id)
            label += f"  {sp:.0f}px/s"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Direction arrow from the box center.
        if show_direction and speed_estimator is not None:
            dx, dy = speed_estimator.direction_of(t.track_id)
            sp = speed_estimator.speed_of(t.track_id)
            if sp > 10.0:
                cx, cy = t.center()
                length = min(60, max(20, sp / 5))
                end = (int(cx + dx * length), int(cy + dy * length))
                cv2.arrowedLine(
                    out,
                    (int(cx), int(cy)),
                    end,
                    color,
                    2,
                    tipLength=0.4,
                )
    return out


def draw_trajectories(
    frame: np.ndarray,
    trails: Dict[int, "deque"],  # noqa: F821
    fade: bool = True,
) -> np.ndarray:
    """Overlay polyline trails for every tracked id.

    Older points in the trail are drawn with lower opacity when ``fade`` is
    True, giving a nice motion-blur-esque aesthetic.
    """
    out = frame.copy()
    for tid, trail in trails.items():
        if len(trail) < 2:
            continue
        color = color_for_id(tid)
        points = list(trail)
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            if fade:
                alpha = i / len(points)
                c = tuple(int(ch * alpha) for ch in color)
            else:
                c = color
            cv2.line(out, pt1, pt2, c, 2, cv2.LINE_AA)
    return out
