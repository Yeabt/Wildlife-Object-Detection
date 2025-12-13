import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile

from utils.load_model import load_models
from utils.inference_yolo import run_inference_yolo_image, run_inference
from utils.inference_rtdetr import run_inference_rtdetr_image

st.set_page_config(page_title="Multi-Model Object Detection", layout="wide")

st.title("🦌 Wildlife Object Detection")

# Load models
yolo_model, rtdetr_model = load_models()

# Sidebar controls
st.sidebar.header("Settings")

conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

mode = st.radio(
    "Select Mode",
    ["Image", "Video", "Webcam", "Model Comparison"],
    horizontal=True
)

if mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        model_choice = st.selectbox("Select Model", ["YOLO", "RT-DETR"])
        if model_choice == "YOLO":
            model = yolo_model
            with st.spinner("Running inference..."):
                output_img, results = run_inference_yolo_image(model, image_np, conf)
        else:
            model = rtdetr_model
            with st.spinner("Running inference..."):
                output_img, results = run_inference_rtdetr_image(model, image_np, conf)

        col1, col2 = st.columns(2)

        col_width = 1000

        with col1:
            st.subheader("Original Image")
            st.image(image_np, width=col_width)

        with col2:
            st.subheader(f"Detected ({model_choice})")
            st.image(output_img, width=col_width)

        st.subheader("Detections")
        st.write(results.boxes.data.cpu().numpy())

    
if mode == "Video":
    st.subheader("🎥 Video Inference")

    model_choice = st.selectbox("Select Model", ["YOLO", "RT-DETR"])
    model = yolo_model if model_choice == "YOLO" else rtdetr_model

    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if model_choice == "YOLO":
                output, fps, _ = run_inference(model, frame, imgsz=640)
            else:
                output, fps, _ = run_inference(model, frame, imgsz=512)

            stframe.image(
                output,
                caption=f"{model_choice} | FPS: {fps:.2f}",
                width=1000
            )

        cap.release()

if mode == "Model Comparison":
    st.subheader("🆚 YOLO vs RT-DETR Comparison")

    uploaded_file = st.file_uploader(
        "Upload an image for comparison",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with st.spinner("Running YOLO..."):
            yolo_out, yolo_fps, _ = run_inference(
                yolo_model, image_np, conf=conf
            )

        with st.spinner("Running RT-DETR..."):
            rtdetr_out, rtdetr_fps, _ = run_inference(
                rtdetr_model, image_np, conf=conf, imgsz=512
            )

        col1.subheader(f"YOLO (FPS: {yolo_fps:.2f})")
        col1.image(yolo_out, width=1000)

        col2.subheader(f"RT-DETR (FPS: {rtdetr_fps:.2f})")
        col2.image(rtdetr_out, width=1000)

        st.markdown("### 📊 Performance Summary")
        st.table({
            "Model": ["YOLO", "RT-DETR"],
            "FPS": [round(yolo_fps, 2), round(rtdetr_fps, 2)]
        })

if mode == "Webcam":
    st.subheader("📸 Webcam Inference")

    model_choice = st.selectbox("Select Model", ["YOLO", "RT-DETR"])
    model = yolo_model if model_choice == "YOLO" else rtdetr_model

    conf = st.slider("Confidence", 0.1, 1.0, 0.25)

    camera_image = st.camera_input("Take a photo")

    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Running inference..."):
            output, fps, _ = run_inference(model, image_np)

        st.image(
            output,
            caption=f"{model_choice} | FPS: {fps:.2f}",
            width=1000
        )
