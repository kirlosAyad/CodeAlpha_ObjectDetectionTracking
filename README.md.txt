# Object Detection and Tracking with YOLOv8 and SORT

## Overview

This project performs real-time object detection and tracking using the YOLOv8 model combined with the SORT (Simple Online and Realtime Tracking) algorithm.

- **Object Detection:** Uses YOLOv8 (You Only Look Once, version 8) for fast and accurate detection.
- **Object Tracking:** Uses SORT to assign IDs to detected objects and track their movement across frames.

## Features

- Real-time video capture from webcam.
- Detects objects with confidence threshold filtering.
- Tracks multiple objects with unique IDs.
- Visualizes bounding boxes and track IDs on video frames.

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Ultralytics (YOLOv8)
- SciPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kirlosAyad/CodeAlpha_ObjectDetectionTracking.git
   cd CodeAlpha_ObjectDetectionTracking

2.Install dependencies:

```bash
pip install -r requirements.txt

