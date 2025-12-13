import cv2
import numpy as np
import time

def run_inference_yolo_image(model, image, conf=0.25):
    results = model.predict(
        source=image,
        conf=conf,
        imgsz=640,
        verbose=False,
        device=0 if model.device.type == "cuda" else "cpu"
    )
    
    annotated = results[0].plot()
    return annotated, results[0]

def run_inference(model, frame, conf=0.25, imgsz=640):
    start = time.time()

    results = model.predict(
        source=frame,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
        device="cuda" if model.device.type == "cuda" else "cpu"
    )

    end = time.time()
    fps = 1 / (end - start)

    annotated = results[0].plot()
    return annotated, fps, results[0]