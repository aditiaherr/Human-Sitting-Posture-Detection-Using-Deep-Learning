# Real-Time Human Sitting Posture Detection using YOLOv5

## Introduction

Prolonged sitting, especially in poor posture, is one of the leading causes of musculoskeletal disorders and spinal problems in modern workplaces. As individuals increasingly spend extended hours at desks, there is a pressing need for intelligent systems that can monitor and promote healthy posture habits. This project addresses this need by developing a real-time posture detection system using YOLOv5 and OpenCV, capable of identifying whether a user's sitting posture is "Good" or "Bad" and providing live feedback.

This system can be deployed using a standard webcam and does not require any external sensors or special hardware, making it a scalable, low-cost, and non-invasive solution for improving workplace ergonomics.

![Bad Sitting Posture](https://github.com/aditiaherr/Human-Sitting-Posture-Detection-Using-Deep-Learning/blob/main/Sitting_Bad.png?raw=true)

## Problem Statement

Despite the availability of ergonomic chairs and awareness campaigns, users often fail to maintain a proper sitting posture consistently. Current commercial solutions like smart chairs and posture trackers are either expensive or lack real-time feedback mechanisms. There is a lack of accessible, intelligent tools that can assist users in correcting their sitting habits dynamically and continuously.

This project aims to build a computer vision-based application that can:
- Detect a person in a sitting position using webcam input.
- Classify their posture as "Good" or "Bad".
- Provide real-time visual feedback through bounding boxes.
- Send periodic posture and break reminders to encourage healthier sitting behavior.

## Project Objective

- Build a posture classification system using YOLOv5 object detection.
- Use a dataset that simulates realistic office/home environments.
- Achieve high accuracy, precision, and recall for classification.
- Deploy the model into a system that processes live video input and responds with annotations and alerts.

## Working Overview

1. **Image Input**: The system captures real-time video using a webcam.
2. **Object Detection**: YOLOv5 is used to detect human figures in sitting positions.
3. **Posture Classification**: Based on bounding box region and trained data, the posture is classified as either "Good" or "Bad".
4. **Visual Feedback**: Bounding boxes are color-coded (Green for Good, Red for Bad) and displayed on the video frame.
5. **Reminders**: Every 15 minutes, reminders such as “Take a break” or “Maintain good posture” are displayed.

## Technical Details

### Dataset

- Total images: 1,342
  - Good posture: 702 images
  - Bad posture: 640 images
- Data Source: Collected and labeled via Roboflow
- Preprocessing:
  - Resized to 640x640 pixels
  - Normalization and augmentation applied
  - Real-world scenes with background noise
- Splits:
  - Training: 70%
  - Validation: 20%
  - Testing: 10%

### Model

- Architecture: YOLOv5s
- Training environment: Google Colab with NVIDIA T4 GPU
- Number of epochs: 200 (initial) + 100 (fine-tuning)
- Batch size: 25
- Transfer Learning: Used pretrained YOLOv5 weights
- Input format: COCO YAML configuration with 2 classes – "Good" and "Bad"

### Tools and Libraries

- YOLOv5 (PyTorch)
- OpenCV
- NumPy
- Google Colab
- Roboflow
- Flask (optional for front-end)

### System Pipeline

1. **Data Labeling** using Roboflow
2. **Model Training** with YOLOv5 on Google Colab
3. **Model Evaluation** using mAP, precision, recall, F1-score
4. **Live Inference** using OpenCV to draw results on real-time video

## Results

| Metric        | Good Posture | Bad Posture | Overall |
|---------------|--------------|-------------|---------|
| Precision     | 83.2%        | 65.6%       | 74.4%   |
| Recall        | 77.5%        | 81.9%       | 79.7%   |
| mAP@0.5       | 72.8%        | 85.0%       | 78.9%   |
| mAP@0.5:0.95  | 51.7%        | 55.6%       | 53.6%   |
| Inference Time| -            | -           | 1.36s   |

- The system performs better in identifying “Bad” postures due to their distinct visual cues.
- F1-confidence and precision-recall curves show stable performance across confidence thresholds.
- Confusion matrix highlights strong classification ability with minor misclassification between background and posture classes.

## System Architecture

- **Backbone**: Extracts features from input images.
- **Neck**: Aggregates features at different resolutions.
- **Head**: Outputs bounding boxes and class scores.
- **OpenCV Pipeline**: Captures video, processes frame-by-frame, and visualizes results.

The architecture is optimized to process real-time data with minimal latency and can be deployed on a local machine with standard specifications.

## Visual Output Examples

- Real-time detection with bounding boxes labeled "Good" and "Bad"
- Screenshots of detection on multi-user frames
- Precision and Recall plots
- Confusion Matrix
- Loss vs Epochs graphs

## Limitations and Challenges

- The model is currently trained only on static sitting postures; dynamic movements (e.g., leaning forward then correcting) are not detected.
- Some false positives in multi-person detection when users are partially visible.
- Real-time processing may vary based on webcam resolution and system specs.

## Future Scope

- Support dynamic posture tracking over time using temporal models (e.g., LSTM, ST-GCN).
- Integrate wearable sensors for hybrid vision-sensor approach.
- Develop mobile and web-based deployment using TensorFlow Lite or ONNX.
- Add user-specific posture tracking dashboard and analytics.
- Incorporate reinforcement learning for personalized feedback.

## How to Run

1. Clone the repository:
git clone https://github.com/your-username/real-time-posture-detection-yolov5
cd real-time-posture-detection-yolov5
