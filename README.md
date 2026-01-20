# Real-Time Multi-Object Tracking (MOT) Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-orange?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8-green?style=for-the-badge&logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Deployment-Docker-blue?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

## 🔬 Research Objective
This repository hosts a modular **Multi-Object Tracking (MOT)** system designed for high-fidelity pedestrian and vehicle tracking in complex surveillance scenarios. The core objective of this research was to implement a robust **"Tracking-by-Detection"** pipeline that minimizes ID switching (Identity preservation) even during extended periods of occlusion.

The system integrates state-of-the-art object detectors (YOLOv8) with motion estimation algorithms (Kalman Filters) and data association techniques (Hungarian Algorithm/SORT) to achieve real-time performance (>30 FPS) on edge devices.

## 🛠️ System Architecture

### 1. The Pipeline
The tracking logic follows a deterministic three-stage process:
1.  **Detection:** A deep neural network (YOLOv8/YOLOv5) infers bounding boxes from the raw video frame.
2.  **State Estimation:** A **Kalman Filter** predicts the future centroid of each object based on its velocity vector.
3.  **Data Association:** The Hungarian Algorithm matches the predicted centroids with the new detections using IoU (Intersection over Union) distance metrics, assigning unique IDs to distinct objects.

### 2. Key Features
* **Occlusion Handling:** Robust re-identification logic to maintain IDs when objects cross paths.
* **Modular Backbone:** Supports swappable detection heads (YOLO, EfficientDet).
* **Pedestrian Extraction:** Dedicated scripts to isolate and export individual pedestrian snapshots for Re-ID datasets.

## 📂 Repository Structure & Modules

The codebase is organized into modular components to separate detection, tracking, and utility logic.

| Directory / Module | Description |
| :--- | :--- |
| **`tracker/`** | **Core Logic:** Contains the tracking algorithms (SORT/DeepSORT), Kalman Filter implementations, and linear assignment cost matrix functions. |
| **`cfg/`** | **Configuration:** YAML files defining model hyperparameters, IoU thresholds, and tracking confidence limits. |
| **`models/`** | **Network Architecture:** Definitions for the neural network backbones and feature extractors. |
| **`utils/`** | **Helper Functions:** Visualization tools (bounding box drawing), logger setup, and coordinate geometry conversions. |
| **`data/`** | **Input/Output:** Default directory for storing raw video samples and pre-trained weights (`.pt` / `.weights`). |
| **`docker/`** | **Containerization:** Dockerfiles and build scripts for deploying the tracker in isolated environments. |
| **`assets/`** | **Media:** Test images and gifs used for documentation and regression testing. |
| `demo.py` | **Inference Script:** The main entry point to run tracking on video files or webcam streams. |
| `extract_ped_per_frame.py` | **Data Mining:** A utility script to crop and save every tracked person as a separate image (useful for training Re-ID models). |

## 🚀 Quick Start

### Prerequisites
* Python 3.8+
* PyTorch (CUDA recommended for GPU acceleration)
* OpenCV

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/HarshaVardhan3002/Object_tracker.git](https://github.com/HarshaVardhan3002/Object_tracker.git)
    cd Object_tracker
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Tracking Inference:**
    ```bash
    # Run on a local video file
    python demo.py --source data/video_sample.mp4 --save-vid

    # Run on Webcam
    python demo.py --source 0
    ```

### 🧠 Custom Training
To train the underlying detector on a custom dataset (e.g., VisDrone or MOT17):
```bash
python train.py --data cfg/custom_data.yaml --epochs 50 --batch-size 16
