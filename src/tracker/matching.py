"""Cost matrices and Hungarian-algorithm assignment for data association."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, bbox_a[2] - bbox_a[0]) * max(0.0, bbox_a[3] - bbox_a[1])
    area_b = max(0.0, bbox_b[2] - bbox_b[0]) * max(0.0, bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_distance(tracks_xyxy: Sequence[np.ndarray], dets_xyxy: Sequence[np.ndarray]) -> np.ndarray:
    """Cost matrix (rows=tracks, cols=detections) using 1 - IoU."""
    if len(tracks_xyxy) == 0 or len(dets_xyxy) == 0:
        return np.zeros((len(tracks_xyxy), len(dets_xyxy)), dtype=np.float32)
    cost = np.zeros((len(tracks_xyxy), len(dets_xyxy)), dtype=np.float32)
    for i, t in enumerate(tracks_xyxy):
        for j, d in enumerate(dets_xyxy):
            cost[i, j] = 1.0 - iou(t, d)
    return cost


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian assignment with a threshold.

    Returns:
        matches:        list of (track_idx, detection_idx) pairs.
        unmatched_a:    track indices with no assignment.
        unmatched_b:    detection indices with no assignment.
    """
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((int(r), int(c)))
            matched_a.add(int(r))
            matched_b.add(int(c))

    unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_a]
    unmatched_b = [i for i in range(cost_matrix.shape[1]) if i not in matched_b]
    return matches, unmatched_a, unmatched_b
