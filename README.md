# Pose-Estimation-using-YOLO-MediaPipe
This repository contains the Yolo_media_streamlit.py application, which integrates YOLO for person detection and MediaPipe for pose estimation.

Features

YOLO-based Person Detection

MediaPipe-based Pose Estimation

Real-time Processing on Images, Videos, and Live Webcam

Download Processed Videos

Streamlit UI for Easy Interaction

Requirements

Ensure you have the following dependencies installed:

pip install streamlit opencv-python numpy ffmpeg-python mediapipe

Running the Application

To run the YOLO + MediaPipe-based pose estimation app:

streamlit run Yolo_media_streamlit.py

Model Files Required

Download the YOLO model weights and config before running:

wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights
wget https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg

Place them in the project directory.

Usage

Upload an image or video for processing.

Select live webcam to process real-time pose estimation.

Download processed video after analysis.

License

This project is open-source under the MIT License.
