# 🦌 Multi-Model Object Detection & Tracking with Streamlit

This project is a computer vision web application that deploys and compares YOLO and RT-DETR object detection models using Streamlit.
It supports image inference, video inference, webcam capture, and multi-object tracking with ByteTrack, enabling real-time-style wildlife monitoring and performance comparison.

# ✨ Key Features
- 🔍 Image Object Detection
- 🎥 Video Inference
- 📸 Webcam Snapshot Inference
- 🆚 Side-by-Side Model Comparison (YOLO vs RT-DETR)
- 📊 FPS Performance Benchmarking
- 🖥️ Interactive Web UI (Streamlit)

# 🧠 Models Used
| Model         | Description                                      |
| ------------- | ------------------------------------------------ |
| **YOLO**      | Fast, real-time object detection                 |
| **RT-DETR**   | Transformer-based detector with global reasoning |

# 📸 Dataset used
- [CCT20 Benchmark subset] https://lila.science/datasets/caltech-camera-traps

# 📁 Project Structure
```
project/
│
├── models/
│   ├── yolo.pt
│   └── rtdetr.pt
│
├── utils/
│   ├── load_model.py
│   └── inference.py
│
├── app.py
├── requirements.txt
└── README.md
```

# ⚙️ Installation
## 1️⃣ Create Environment
```
conda create -n cv_app python=3.10
conda activate cv_app
```

## 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

## 🚀 Run the Application
```
streamlit run app.py
```

## ✨ Credits
- [Ultralytics YOLO11] https://docs.ultralytics.com/models/yolo11/#how-do-i-train-a-yolo11-model-for-object-detection
- [Baidu's RT-DETR: A Vision Transformer-Based Real-Time Object Detector] https://docs.ultralytics.com/models/rtdetr/#overview
