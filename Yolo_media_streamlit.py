import streamlit as st
import cv2 as cv
import numpy as np
import tempfile
import os
from datetime import datetime
import ffmpeg
import mediapipe as mp

# Load YOLO model
net = cv.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def detect_person_yolo(frame):
    height, width, _ = frame.shape
    blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences = [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # Person class
                center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

def process_frame(frame):
    people_boxes = detect_person_yolo(frame)
    for (x, y, w, h) in people_boxes:
        person_roi = frame[y:y+h, x:x+w]
        if person_roi.size == 0:
            continue

        rgb_frame = cv.cvtColor(person_roi, cv.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(person_roi, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        frame[y:y+h, x:x+w] = person_roi
    return frame

st.title("Pose Estimation using YOLO + MediaPipe")
option = st.radio("Select Input Type:", ("Image", "Video", "Live Webcam"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv.imdecode(file_bytes, 1)
        output_frame = process_frame(frame)
        st.image(output_frame, channels="BGR")
        st.success("Pose estimation completed for the image.")

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        cap = cv.VideoCapture(tfile.name)
        out_file = "4.mp4"
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(temp_output, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame)
            out.write(processed_frame)
        
        cap.release()
        out.release()

        ffmpeg.input(temp_output).output(out_file, vcodec='libx264', format='mp4').run()
        os.remove(temp_output)
        st.video(out_file)
        st.download_button("Download Processed Video", out_file, file_name="processed_video.mp4")
        st.success("Pose estimation completed for the video.")

elif option == "Live Webcam":
    record = st.checkbox("Record Session")
    temp_video = "recorded_video.avi"
    cap = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(temp_video, fourcc, 20.0, (640, 480))
    
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        if record:
            out.write(processed_frame)
        stframe.image(processed_frame, channels="BGR")
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    
    mp4_output = "recorded_video.mp4"
    ffmpeg.input(temp_video).output(mp4_output, vcodec='libx264', format='mp4').run()
    os.remove(temp_video)
    
    st.success("Recording Completed! Pose estimation completed for the live webcam session.")
    st.download_button("Download Recorded Video", mp4_output, file_name=f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
