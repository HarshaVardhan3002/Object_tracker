# Architecture

This document goes one level deeper than the README: it explains *why* each
component is shaped the way it is, and gives the math behind the Kalman
filter and the ByteTrack-style association so the implementation is fully
traceable from the paper.

## 1. Pipeline overview

```
┌───────────────┐   BGR frame    ┌──────────────┐  Detections  ┌──────────────┐
│ cv2.VideoCapture │ ─────────▶  │ YOLODetector │ ───────────▶ │ ByteTracker  │
└───────────────┘               └──────────────┘              └──────┬───────┘
                                                                      │ Tracks
                                                                      ▼
                                                     ┌──────────────────────────┐
                                                     │ Analytics (counts,       │
                                                     │ heatmap, trails, speed)  │
                                                     └──────────────┬───────────┘
                                                                    │
                                                                    ▼
                                                     ┌──────────────────────────┐
                                                     │ Streamlit UI / CLI       │
                                                     └──────────────────────────┘
```

Each boundary is a simple dataclass (`Detection`, `Track`, `FrameResult`), so
the stages are independently swappable. You could drop in RT-DETR for the
detector, DeepSORT for the tracker, or a FastAPI server for the UI without
touching the other modules.

## 2. The detector layer

`src/detector/yolo.py` is a deliberately thin wrapper:

* Constructs a single `ultralytics.YOLO` model and caches it for the life of
  the process — Streamlit reruns should not reload weights.
* Applies a confidence + NMS-IoU gate up-front so the tracker never sees
  low-value detections.
* Maps COCO class ids to names and supports a user-supplied class filter,
  which keeps the UI multi-select lightweight.

I chose to keep the YOLOv8 call at the frame level (not batched) because the
rest of the pipeline is inherently sequential (Kalman predict/update per
frame), so batching only pays off if you have multiple independent cameras.

## 3. The Kalman filter

The state vector is:

```
x = (cx, cy, a, h, vx, vy, va, vh)ᵀ
```

where `(cx, cy)` is the bounding-box center, `a = w/h` the aspect ratio,
`h` the box height, and `v*` the corresponding velocities.

**Motion model** (constant velocity, `dt = 1` frame):

```
F = | I₄  I₄ |
    |  0  I₄ |
```

so `x_{k+1|k} = F x_{k|k}` and the position component simply drifts by its
velocity each frame.

**Observation model** — we only observe `(cx, cy, a, h)`:

```
H = [ I₄  0 ]
```

**Noise covariances** scale with the current box height `h`, which is the
same parameterization used in SORT/DeepSORT/ByteTrack. The intuition: a car
50 m away is only a few pixels tall, so absolute pixel noise should be
tighter than for a car that fills half the frame.

The Kalman gain is computed via a Cholesky factorization
(`scipy.linalg.cho_factor` / `cho_solve`) rather than a direct inverse for
numerical stability when the covariance matrix becomes close to singular
during long occlusions.

## 4. ByteTrack-style two-stage association

Vanilla SORT throws away detections below some confidence threshold, which
is exactly when you most want them — during partial occlusion, detector
confidence dips even when the object is still partially visible.

ByteTrack (Zhang et al., 2022) keeps those low-confidence detections for a
second round of association, which dramatically reduces identity switches
at a near-zero compute cost.

The concrete algorithm implemented in `src/tracker/byte_tracker.py`:

1. **Predict** all active and recently-lost tracks one step forward.
2. **Split** this frame's detections by confidence:
   * `high_dets ≥ high_thresh`  (default 0.5)
   * `low_thresh ≤ low_dets < high_thresh`
3. **Stage 1:** build the `1-IoU` cost matrix between (`tracked ∪ lost`) and
   `high_dets`, mask out cross-class pairs, run the Hungarian algorithm,
   reject matches with cost > `match_thresh`.
4. **Stage 2:** for tracks that didn't find a partner in stage 1 and are
   *currently tracked*, run a second Hungarian pass against `low_dets` with
   a tighter threshold (0.5). This is the ByteTrack "second chance".
5. **New tracks** come from unmatched `high_dets`; unmatched low-confidence
   detections are discarded (as in the original paper).
6. **Aging** — a track that has gone more than `max_lost_frames` without an
   update is permanently removed.

Keeping the association class-aware (cross-class pairs are masked out in the
cost matrix) is a small but meaningful quality win over vanilla SORT, which
happily reassigns a pedestrian ID to an overlapping vehicle.

## 5. Analytics

Each analytics module consumes the same `Track` list and is stateful
across frames:

* **`ObjectCounter`** keeps a `Dict[class_name, set[track_id]]`; a track id
  is only counted once for "unique IDs" even if it disappears and a new id
  later reappears in the same spot.
* **`MotionHeatmap`** stamps a Gaussian-blurred hot spot at each track
  center into a running 2-D accumulator, with a slow geometric decay so
  the heatmap stays responsive to changes in scene activity.
* **`TrajectoryStore`** is a bounded `deque` per track id — the
  trail renderer fades older points so the visual stays uncluttered.
* **`SpeedEstimator`** uses a short sliding window of centers (default 8
  frames) and computes the mean displacement per frame, converted to
  pixels-per-second using the source video FPS. Multiply by a calibration
  factor to get m/s if you have a known real-world reference.

## 6. UI

Streamlit was chosen over Flask/FastAPI + React because:

1. The whole app is one `app.py` — easy to skim, easy to deploy.
2. `st.session_state` keeps the tracker hyperparameters across reruns,
   which is how every widget change works in Streamlit.
3. The `.streamlit/config.toml` theme + a small amount of injected CSS
   (metric cards with a gradient border) is enough to make it look
   intentional rather than "just a Streamlit app".

For production use where multiple viewers share a single stream you'd swap
this for a WebRTC server; the `TrackingPipeline` class would not change.

## 7. Testing strategy

Tests in `tests/test_tracker.py` live entirely in numpy-land:

* IoU / assignment correctness on small hand-built matrices.
* End-to-end id persistence on a synthetic diagonally-moving object.
* Class-awareness — a same-location car and person must get different ids.
* Aging-out behavior after `max_lost_frames` frames of no updates.
* Analytics correctness (counter, heatmap accumulation, speed sign).

Because no YOLO weights are required, the whole suite runs in
sub-second time in CI on GitHub Actions.
