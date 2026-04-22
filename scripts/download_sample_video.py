"""Download a small sample video so you can demo the tracker without hunting
for footage. Saves to ``data/sample.mp4``.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


# A short, permissively-licensed pedestrian clip hosted by Ultralytics for demos.
SAMPLE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/people.mp4"
TARGET = Path("data/sample.mp4")


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SAMPLE_URL} -> {TARGET} …")
    try:
        urllib.request.urlretrieve(SAMPLE_URL, TARGET)
    except Exception as exc:  # pragma: no cover
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Saved {TARGET} ({TARGET.stat().st_size/1e6:.1f} MB).")


if __name__ == "__main__":
    main()
