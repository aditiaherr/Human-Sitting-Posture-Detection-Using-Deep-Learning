# Human Sitting Posture Detection Using Deep Learning

This project aims to develop an efficient system for analyzing human sitting postures from uploaded images and categorizing them as either good or bad posture. It utilizes a combination of YOLOv5 for object detection and Convolutional Neural Networks (CNNs) for posture classification. The process involves collecting and labeling a diverse dataset of sitting posture images, training the model, and evaluating its performance. The system will provide feedback to users on their sitting posture, fostering awareness and potentially leading to improved health outcomes. Future enhancements may include capturing image or video through webcam and giving real-time feedback, integration with mobile apps for on-the-go posture monitoring, and additional features like posture correction suggestions.

## Abstract

Human posture detection is crucial in various applications such as healthcare, human-computer interaction, and workplace ergonomics. Detecting human sitting posture can help facilitate posture correction and improve workplace ergonomics. This study focuses on detecting human sitting posture using the YOLOv5 model for object detection. A dataset of sitting posture images is annotated and used to train the YOLOv5 model. The model is evaluated based on precision, recall, and mean average precision (mAP). The system can be integrated into practical applications like posture correction and workplace ergonomics. Future work includes enhancing the model for real-time feedback and low-complexity implementation.

**Keywords**: CNN, Deep learning, Posture detection, YOLOv5.

## Introduction

Various sectors require human sitting posture detection to avoid healthcare issues and improve workplace ergonomics. This paper introduces a deep learning model, YOLOv5, that is popular for its speed and efficiency in object detection tasks. We collected a diverse dataset of sitting postures, annotated the images, and used data augmentation techniques for training. The YOLOv5 model was trained using PyTorch, and its performance was evaluated based on precision, recall, and mAP. The model was then integrated into practical applications, such as user interface development for posture correction tools.

## Problem Statement

Detecting human posture using deep learning models such as YOLOv5 plays a significant role in healthcare and workplace ergonomics. The challenge is to develop a system that can accurately detect and classify various sitting postures, even in different environmental conditions. The solution focuses on the YOLOv5 model, which is known for its speed and accuracy in object detection. The model is trained using a comprehensive dataset of sitting postures, and the system provides feedback on whether the posture is good or bad, contributing to improved health and safety.

## Literature Survey

Several techniques have been explored for human posture detection:

- **Skeleton-based Online Human Activity Recognition (HAR)**: Methods like Spatio-Temporal Graph Convolutional Networks (ST-GCN) have been used to extract spatial and temporal information for accurate human activity prediction.
- **Deep Convolutional Neural Networks (CNN)**: CNNs have been used for posture recognition by extracting depth maps or posture features, achieving high precision in human action identification.
- **YOLOv5 for Real-time Detection**: YOLOv5 is an efficient model for real-time object detection, including human posture detection. It has shown high precision and recall in detecting common postures like sitting, walking, or falling.
- **Evaluation Metrics**: Precision, recall, and mean average precision (mAP) are used to evaluate model performance. These metrics provide insights into the accuracy and reliability of the posture detection system.

## Working Principle

1. **Data Collection and Annotation**: A large dataset of sitting posture images is collected, and annotations are added using a tool like LabelImg. The dataset is augmented with techniques such as rotation, scaling, and flipping to increase robustness.
2. **Model Development and Training**: The YOLOv5 model is trained using the annotated dataset. Data preprocessing, including normalization and resizing, is performed to make the data compatible with YOLOv5 input requirements. The model is trained using PyTorch, with hyperparameters optimized for performance.
3. **Model Evaluation**: The model's performance is evaluated using a validation dataset. Metrics such as precision, recall, and mAP are calculated to assess accuracy. Iterative optimization is done to fine-tune the model.
4. **System Integration**: The trained model is integrated into applications such as posture correction tools, ergonomic assessments, and surveillance systems.

## Result Analysis

The model has been evaluated on a dataset with images of various sitting postures labeled as "Sitting_good" or "Sitting_bad". The results show that the model can effectively detect and classify sitting postures. Key performance metrics include:

- **Precision and Recall**: The precision and recall values are plotted to evaluate the model's accuracy and reliability in detecting postures.
- **F1 Score**: The F1 curve is used to visualize the balance between precision and recall.

![Architecture](https://github.com/aditiaherr/Human-Sitting-Posture-Detection-Using-Deep-Learning/blob/main/architecture.png)

![Results](https://github.com/aditiaherr/Human-Sitting-Posture-Detection-Using-Deep-Learning/blob/main/results.png)

## Challenges and Future Scope

Challenges include the variability of images, lighting conditions, and background noise, which can affect posture detection accuracy. Achieving real-time efficiency is also critical for applications in workplace ergonomics. 

**Future Scope**:
- **Real-time Feedback**: The system could be enhanced to capture live images or videos and provide real-time feedback to users on their posture.
- **Privacy Concerns**: Ensuring data privacy while handling posture data is important, especially when capturing continuous data.
- **Edge Computing**: Deploying the system with edge computing for faster real-time processing and better efficiency.

## Conclusion

The YOLOv5 model for detecting human sitting postures represents a significant advancement in human-computer interaction. By providing real-time feedback on posture, it can promote healthier habits and improve workplace ergonomics. Future work will focus on refining the model, integrating multimodal data, and exploring new application areas.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Human-Sitting-Posture-Detection.git
