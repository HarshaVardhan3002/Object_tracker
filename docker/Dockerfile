## Multi-stage Docker build for VisionTrack.
#
# Usage:
#   docker build -t visiontrack -f docker/Dockerfile .
#   docker run -p 8501:8501 visiontrack
#
# The image is CPU-only by default so it can run anywhere. Swap the base
# image for nvidia/cuda:12.1.1-runtime-ubuntu22.04 to enable GPU inference.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker caches this layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source last so code changes don't bust the deps layer.
COPY . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
