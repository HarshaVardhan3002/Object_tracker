"""Command-line tracker runner.

Examples::

    # Track a local video and save the annotated result next to it.
    python demo.py --source data/sample.mp4 --output results/sample_tracked.mp4

    # Only track people and cars at 480p for speed.
    python demo.py --source data/sample.mp4 --classes person car --imgsz 480

    # Run on a webcam.
    python demo.py --source 0 --show

The script is intentionally small -- all of the real logic lives in
:mod:`src.pipeline`. This makes it easy to drop the same pipeline into a
Streamlit UI, a FastAPI endpoint, or a batch job.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2

from src.pipeline import PipelineConfig, TrackingPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionTrack CLI demo")
    p.add_argument("--source", required=True, help="Path to video file, or webcam index (e.g. 0)")
    p.add_argument("--output", default=None, help="Output video path (mp4). If omitted, no video is written.")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights (auto-downloaded)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--classes", nargs="*", default=None, help="Class names to keep (COCO).")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all).")
    p.add_argument("--show", action="store_true", help="Display a live preview window.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    source: str | int = args.source
    if source.isdigit():
        source = int(source)

    config = PipelineConfig(
        weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        class_filter=args.classes,
    )
    pipeline = TrackingPipeline(config=config)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline.fps = fps
    pipeline.speed.fps = fps

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    i = 0
    tic = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and i >= args.max_frames:
                break

            result = pipeline.step(frame)
            if writer is not None:
                writer.write(result.annotated)
            if args.show:
                cv2.imshow("VisionTrack", result.annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            i += 1
            if i % 30 == 0:
                print(
                    f"[{i:5d}] active={sum(result.counts_current.values()):3d} "
                    f"unique={sum(result.counts_total.values()):3d} "
                    f"mean_speed={result.speed_summary.get('mean_speed', 0):.0f}px/s "
                    f"FPS={i/(time.time()-tic):.1f}"
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(
        f"Done. Processed {i} frames in {time.time()-tic:.1f}s "
        f"({i/max(1e-6, time.time()-tic):.1f} FPS)."
    )
    if args.output:
        print(f"Saved annotated video -> {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
