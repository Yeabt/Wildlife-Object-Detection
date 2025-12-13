import cv2
import numpy as np
import time

def run_inference_rtdetr_image(model, image, conf=0.25):
    results = model.predict(
        source=image,
        conf=conf,
        imgsz=512,
        verbose=False,
        device=0 if model.device.type == "cuda" else "cpu"
    )

    annotated = results[0].plot()
    return annotated, results[0]

