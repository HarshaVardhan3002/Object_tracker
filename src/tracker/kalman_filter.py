"""Constant-velocity Kalman filter for bounding-box tracking.

The state vector is (cx, cy, a, h, vx, vy, va, vh) where:

* cx, cy -- box center
* a      -- aspect ratio (w / h)
* h      -- height
* vx..vh -- their velocities

This is the same parameterization used in the SORT / DeepSORT / ByteTrack
line of work. It is implemented from scratch (no filterpy dependency) so
the math is fully visible and there are no hidden black boxes for reviewers.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """8-state constant-velocity Kalman filter for 2D bounding boxes."""

    # Measurement noise is scaled by the object height -- larger objects
    # tolerate larger measurement errors in absolute pixel terms.
    STD_WEIGHT_POSITION = 1.0 / 20.0
    STD_WEIGHT_VELOCITY = 1.0 / 160.0

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # State transition: next_pos = pos + velocity * dt
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation matrix: we only observe position, not velocity.
        self._update_mat = np.eye(ndim, 2 * ndim)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a new track from an initial (cx, cy, a, h) measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self.STD_WEIGHT_POSITION * measurement[3],
            2 * self.STD_WEIGHT_POSITION * measurement[3],
            1e-2,
            2 * self.STD_WEIGHT_POSITION * measurement[3],
            10 * self.STD_WEIGHT_VELOCITY * measurement[3],
            10 * self.STD_WEIGHT_VELOCITY * measurement[3],
            1e-5,
            10 * self.STD_WEIGHT_VELOCITY * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    # ------------------------------------------------------------------
    # Predict / update
    # ------------------------------------------------------------------
    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self.STD_WEIGHT_POSITION * mean[3],
            self.STD_WEIGHT_POSITION * mean[3],
            1e-2,
            self.STD_WEIGHT_POSITION * mean[3],
        ]
        std_vel = [
            self.STD_WEIGHT_VELOCITY * mean[3],
            self.STD_WEIGHT_VELOCITY * mean[3],
            1e-5,
            self.STD_WEIGHT_VELOCITY * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self.STD_WEIGHT_POSITION * mean[3],
            self.STD_WEIGHT_POSITION * mean[3],
            1e-1,
            self.STD_WEIGHT_POSITION * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean, covariance + innovation_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain via Cholesky for numerical stability.
        chol_factor, lower = _cho_factor(projected_cov)
        kalman_gain = _cho_solve(chol_factor, lower, covariance @ self._update_mat.T)

        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


# ---------------------------------------------------------------------------
# Small wrapping helpers around scipy -- kept local so the file stays self-contained.
# ---------------------------------------------------------------------------
def _cho_factor(a: np.ndarray):
    import scipy.linalg

    return scipy.linalg.cho_factor(a, lower=True, check_finite=False)


def _cho_solve(chol_factor, lower, b):
    import scipy.linalg

    return scipy.linalg.cho_solve((chol_factor, lower), b.T, check_finite=False).T
