# Human Detection and Tracking

A lightweight project for detecting and tracking people in images and live video using the YOLOv8 model.

## Description

This project provides a simple yet effective solution for human detection in:

- **Static Images**

- **Live Webcam Feeds**

It uses a pre-trained YOLOv8 model for detection and includes two main scripts:

- detect_image.py → Detects people in a given image.

- detect_video.py → Detects and tracks people using your webcam.

## Features
- Accurate Human Detection using YOLOv8

- Image Detection via detect_image.py

- Real-time Webcam Detection via detect_video.py

- Bounding Box Visualization for detected humans

- Lightweight & Easy to Run on most systems

## Getting Started
1. Prerequisites:

- **Python: 3.8+**

- **pip: Python package installer**

2. Installation

```bash
# Clone the repository
git clone https://github.com/your-github-name/human-detection-project.git
cd human-detection-project
```

### Install dependencies

´´´bash
pip install -r requirements.txt
´´´

Download the YOLOv8 pre-trained model (yolov8n.pt) and place it in the models/ folder.
Get it here: Ultralytics YOLOv8 Models

### Usage

**Detect in an Image**

```bash
python detect_image.py --source "path/to/image.jpg"
```

**Detect in Live Webcam**

```bash
python detect_video.py --source 0
```

**Note: 0 refers to your default webcam.**

### Technologies Used
- **Python**

- **YOLOv8 (Ultralytics)**

- **OpenCV**

# Author
**Zeynep Türkyılmaz** [GitHub Profile](https://github.com/ZeynepTurkyilmaz/human-detection-project.git)

