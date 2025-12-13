import streamlit as st
from ultralytics import YOLO, RTDETR

@st.cache_resource
def load_models():
    yolo_model = YOLO("models/best_yolo.pt")
    rtdetr_model = RTDETR("models/best_rtdetr.pt")
    return yolo_model, rtdetr_model
