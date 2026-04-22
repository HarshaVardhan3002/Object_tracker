"""VisionTrack Streamlit dashboard.

Run with::

    streamlit run app.py

The dashboard is split into three regions:

* **Sidebar**  -- model / tracker / analytics controls
* **Main**     -- annotated live preview + heatmap side by side
* **Bottom**   -- analytics strip with class counts, speed metrics and a
                  per-class bar chart.

State is kept in ``st.session_state`` so settings survive Streamlit's
re-execution on every widget change.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.detector.yolo import COCO_CLASSES
from src.pipeline import PipelineConfig, TrackingPipeline


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionTrack - Real-Time Multi-Object Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A bit of custom CSS to make the dashboard feel like a product, not a demo.
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 14px 18px;
        color: #e5e7eb;
    }
    .metric-card h4 { margin: 0; font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.8rem; font-weight: 600; margin-top: 4px; }
    .gradient-header {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(col, title: str, value, subtitle: str = "") -> None:
    col.markdown(
        f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <div class="value">{value}</div>
            <div style="font-size:0.75rem;color:#6b7280;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 class='gradient-header'>🎯 VisionTrack</h1>"
    "<p style='color:#9ca3af;margin-top:-10px;'>Real-time multi-object tracking with live analytics -- YOLOv8 + ByteTrack</p>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Model")
    weights = st.selectbox(
        "YOLOv8 variant",
        options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="Nano is fastest, medium is most accurate. The file auto-downloads on first run.",
    )
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou = st.slider("NMS IoU threshold", 0.1, 0.9, 0.5, 0.05)
    imgsz = st.select_slider("Inference size", options=[320, 480, 640, 960], value=640)

    st.subheader("Class filter")
    default_classes = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]
    classes = st.multiselect(
        "Track only these classes",
        options=COCO_CLASSES,
        default=default_classes,
    )

    st.subheader("Tracker")
    high_thresh = st.slider("High-conf threshold", 0.3, 0.9, 0.5, 0.05)
    low_thresh = st.slider("Low-conf threshold", 0.0, 0.4, 0.1, 0.05)
    match_thresh = st.slider("IoU match threshold", 0.3, 0.95, 0.8, 0.05)
    max_lost = st.slider("Max lost frames (occlusion tolerance)", 5, 120, 30, 5)

    st.subheader("Overlays")
    show_boxes = st.checkbox("Bounding boxes + IDs", True)
    show_trails = st.checkbox("Trajectory trails", True)
    show_heatmap = st.checkbox("Heatmap overlay on video", False)
    trail_length = st.slider("Trail length (frames)", 10, 200, 60, 10)

    st.subheader("Performance")
    stride = st.slider(
        "Process every Nth frame",
        1, 10, 1,
        help="Increase to speed up demo on CPU; tracker still interpolates.",
    )
    max_frames = st.number_input("Max frames to process (0 = all)", 0, 5000, 600, 50)


config = PipelineConfig(
    weights=weights,
    conf_threshold=conf,
    iou_threshold=iou,
    imgsz=imgsz,
    class_filter=classes if classes else None,
    high_thresh=high_thresh,
    low_thresh=low_thresh,
    match_thresh=match_thresh,
    max_lost_frames=int(max_lost),
    show_boxes=show_boxes,
    show_trails=show_trails,
    show_heatmap_overlay=show_heatmap,
    trail_length=int(trail_length),
)


# ---------------------------------------------------------------------------
# Input source
# ---------------------------------------------------------------------------
st.subheader("📹 Input source")
tab_upload, tab_sample = st.tabs(["Upload a video", "Sample demo"])

video_path = None
with tab_upload:
    uploaded = st.file_uploader(
        "Drop an MP4 / MOV / AVI file",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        video_path = tmp.name
        st.success(f"Loaded **{uploaded.name}** ({os.path.getsize(video_path)/1e6:.1f} MB)")

with tab_sample:
    st.caption(
        "The sample GIFs from the original MOT datasets live in the `assets/` folder. "
        "Use them as a quick sanity check."
    )
    for gif in ["assets/MOT16-03.gif", "assets/MOT16-14.gif", "assets/IMG_0055.gif"]:
        if Path(gif).exists():
            st.image(gif, caption=gif, use_container_width=True)


run_btn = st.button(
    "▶️ Start tracking",
    type="primary",
    disabled=video_path is None,
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Main layout: placeholders we will fill in while processing
# ---------------------------------------------------------------------------
metric_slots = st.columns(4)
metric_placeholders = {
    "fps": metric_slots[0].empty(),
    "total": metric_slots[1].empty(),
    "active": metric_slots[2].empty(),
    "mean_speed": metric_slots[3].empty(),
}

preview_col, heat_col = st.columns([3, 2])
video_slot = preview_col.empty()
heat_slot = heat_col.empty()
progress_slot = st.empty()

chart_col, table_col = st.columns([3, 2])
chart_slot = chart_col.empty()
table_slot = table_col.empty()


# Helper to push an analytics frame
def _render(result, frame_idx: int, total: int, fps_live: float) -> None:
    annotated_rgb = cv2.cvtColor(result.annotated, cv2.COLOR_BGR2RGB)
    heat_rgb = cv2.cvtColor(result.heatmap_only, cv2.COLOR_BGR2RGB)
    video_slot.image(annotated_rgb, channels="RGB", use_container_width=True)
    heat_slot.image(heat_rgb, channels="RGB", use_container_width=True, caption="Motion heatmap")

    metric_placeholders["fps"].markdown(
        f"<div class='metric-card'><h4>Live FPS</h4><div class='value'>{fps_live:.1f}</div>"
        f"<div style='font-size:0.75rem;color:#6b7280;'>frame {frame_idx}/{total}</div></div>",
        unsafe_allow_html=True,
    )
    total_unique = sum(result.counts_total.values())
    active = sum(result.counts_current.values())
    metric_placeholders["total"].markdown(
        f"<div class='metric-card'><h4>Unique IDs seen</h4><div class='value'>{total_unique}</div>"
        f"<div style='font-size:0.75rem;color:#6b7280;'>cumulative</div></div>",
        unsafe_allow_html=True,
    )
    metric_placeholders["active"].markdown(
        f"<div class='metric-card'><h4>Active tracks</h4><div class='value'>{active}</div>"
        f"<div style='font-size:0.75rem;color:#6b7280;'>this frame</div></div>",
        unsafe_allow_html=True,
    )
    mean_sp = result.speed_summary.get("mean_speed", 0.0)
    max_sp = result.speed_summary.get("max_speed", 0.0)
    metric_placeholders["mean_speed"].markdown(
        f"<div class='metric-card'><h4>Mean speed</h4><div class='value'>{mean_sp:.0f} px/s</div>"
        f"<div style='font-size:0.75rem;color:#6b7280;'>max {max_sp:.0f}</div></div>",
        unsafe_allow_html=True,
    )

    if result.counts_total:
        df = pd.DataFrame(
            {
                "class": list(result.counts_total.keys()),
                "unique_ids": list(result.counts_total.values()),
                "active_now": [result.counts_current.get(k, 0) for k in result.counts_total],
            }
        ).sort_values("unique_ids", ascending=False)
        chart_slot.bar_chart(df.set_index("class")["unique_ids"], use_container_width=True)
        table_slot.dataframe(df, use_container_width=True, hide_index=True)

    if total > 0:
        progress_slot.progress(min(1.0, frame_idx / max(1, total)))


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
if run_btn and video_path is not None:
    with st.spinner("Loading YOLOv8 weights and warming up…"):
        pipeline = TrackingPipeline(config=config)

    # We manually drive the capture so we can honor the "process every Nth frame"
    # option and update the UI in real time.
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline.fps = src_fps
    pipeline.speed.fps = src_fps
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix="_out.mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None

    frame_idx = 0
    processed = 0
    tic = time.time()
    last_result = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and processed >= max_frames:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            result = pipeline.step(frame)
            last_result = result

            if writer is None:
                h, w = result.annotated.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, src_fps / max(1, stride), (w, h))
            writer.write(result.annotated)

            elapsed = max(1e-6, time.time() - tic)
            fps_live = (processed + 1) / elapsed
            _render(result, frame_idx, total_frames, fps_live)

            processed += 1
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    progress_slot.empty()
    st.success(f"Processed {processed} frames in {time.time() - tic:.1f}s.")

    if last_result is not None:
        st.subheader("⬇️ Download the annotated video")
        with open(out_path, "rb") as f:
            st.download_button(
                "Download MP4",
                data=f,
                file_name="visiontrack_output.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
elif video_path is None:
    st.info("Upload a video above and hit **Start tracking** to see the tracker in action.")
