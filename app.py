import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from model import DeepFakeDetector
import tempfile
import streamlit.components.v1 as components
import os

# Load model
model = DeepFakeDetector()
model.load_state_dict(torch.load(r"deepfake_detector.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract frames
def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        frames.append(preprocess(pil_image))
    cap.release()
    return torch.stack(frames)

# Custom HTML video player using components
def play_video_with_html(temp_path):
    video_html = f"""
        <video width="100%" height="auto" controls>
            <source src="file://{temp_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    """
    components.html(video_html, height=400)

# Custom CSS for dark blue gradient background and styled boxes
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .fake {
        color: #ff4d4d;
        border: 2px solid #ff4d4d;
    }
    
    .real {
        color: #4dff4d;
        border: 2px solid #4dff4d;
    }
    
    .title {
        color: white;
        text-align: center;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: 1px;
    }
    
    .uploader {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# UI
st.markdown('<div class="title"> DeepFake Detector </div>', unsafe_allow_html=True)

with st.container():
    # st.markdown('<div class="uploader">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi"], key="uploader")
    # st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Read the video bytes once
    video_bytes = uploaded_file.read()

    # Show video using Streamlit's built-in player
    st.video(video_bytes)

    # Save video to temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    # Extract frames and run detection
    frames = extract_frames(tmp_path, max_frames=30)
    frames = frames.unsqueeze(0)

    with torch.no_grad():
        output = model(frames)
        prediction = torch.sigmoid(output)

    # Show prediction
    if prediction.item() > 0.7:
        st.markdown(
            """
            <div class="prediction-box fake">
                DEEPFAKE DETECTED!
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="prediction-box real">
                 REAL VIDEO
            </div>
            """, 
            unsafe_allow_html=True
        )
