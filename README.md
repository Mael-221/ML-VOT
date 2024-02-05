# EPITA Multi-Object Tracking System (TP1-TP4)

## Overview

This repository consists of a series of practical exercises designed to teach the principles of multi-object tracking (MOT). The exercises progress from basic object detection (TP1) to sophisticated tracking using IoU, the Hungarian algorithm (TP3), and the Kalman Filter (TP4).

## Running the Tracking System

### Without Kalman Filter

To run the IoU tracker with the Hungarian algorithm but without the Kalman Filter:

```bash
python IoU_tracker.py
```

This script processes the input images and detection data to track objects across frames using the IoU-based method and the Hungarian algorithm for assignment. It generates:

- A set of bounding boxes for detected objects.
- Tracks that associate detections across frames.
- A CSV output file with tracking results, formatted similarly to the ground truth data.

### With Kalman Filter

To run the IoU tracker enhanced with the Kalman Filter:

```bash
python Kalman_IoU.py
```

This script extends the functionality of `IoU_tracker.py` by incorporating a Kalman Filter for each track, providing better prediction and updating of object states, especially in cases of occlusion or noisy detections. It generates:

- Predicted states for each object using the Kalman Filter.
- Updated tracks with more accurate positioning.
- A similar CSV output file with enhanced tracking results.

## TP Summaries

- **TP1**: Introduction to the basics of object detection.
- **TP2**: Development of a simple IoU-based tracker.
- **TP3**: Integration of the Hungarian algorithm into the IoU tracker for optimized assignment.
- **TP4**: Enhancement of the tracker with the Kalman Filter for state estimation.

## Installation and Dependencies

Make sure to install the required Python packages:

```bash
pip install opencv-python numpy scipy tqdm
```

## Usage

Run the main script corresponding to the desired tracking process. The output will be a visual display of the tracking as well as a saved CSV file containing the tracking results.

## Project Structure

- `Detector.py`: Contains the object detection logic.
- `KalmanFilter.py`: Defines the Kalman Filter class.
- `IoU_tracker.py`: Main script for IoU tracking without Kalman Filter.
- `Kalman_IoU.py`: Combines IoU tracking with the Kalman Filter and Hungarian algorithm.
- `utils.py`: Includes utility functions for data loading and frame processing.


