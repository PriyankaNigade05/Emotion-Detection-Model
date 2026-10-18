# Real-Time & Image-Based Emotion Detection System

A lightweight, production-ready web application and computer vision pipeline that utilizes **Flask**, **OpenCV**, and **DeepFace** (backed by Deep Learning frameworks) to detect and analyze human facial emotions either from a live webcam feed or via static image uploads.

---

## 🚀 Key Features

*   **Static Image Upload Portal:** Upload `.jpg` or `.png` facial images through a web interface to receive instantaneous emotion predictions (`source 1`).
*   **Live Webcam Analytics:** Real-time stream processing via OpenCV to dynamically display bounding labels of dominant emotions directly onto the video overlay (`source 1`, `source 3`).
*   **Robust Evaluation Base:** Developed alongside rigorous testing workflows (including automated dataset evaluation matrices with validation accuracies plotted up to 55%+ across challenging FER subsets) (`source 2`).

---

## 📂 Project Architecture

```text
├── app.py                  # Main Flask Web Application Gateway
├── capture_script.py       # Standalone Desktop Webcam Capture Utility
├── requirements.txt        # Core System Dependencies
├── static/
│   └── uploads/            # Temporary storage for processed images
└── templates/
    └── index.html          # UI Dashboard (handles uploads/displays analytics)
```

---

## 🛠️ Prerequisites & Installation

### 1. Environment Setup
Ensure you have Python 3.8+ installed on your local environment. Clone this repository and navigate to its root directory.

### 2. Install Dependencies
Install all necessary packages via `pip`. This project explicitly pins down `tf-keras` alongside `deepface` to ensure stable underlying model compilation:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage & Execution

### Option 1: Running the Flask Web Application
To spin up the web client dashboard, run:
```bash
python app.py
```
*   Open your browser and navigate to `http://127.0.0`.
*   **`/` (Home):** Access the graphical user interface to upload local image files.
*   **`/predict`:** The endpoint handling deep learning inference on your uploaded assets.
*   **`/webcam`:** Instantiates a programmatic OpenCV session using your hardware's index `0` camera array. Press `q` or `ESC` to terminate the socket thread safely.

### Option 2: Standalone Snapshot Capture Utility
If you prefer running a direct desktop window frame script outside of the web server pipeline:
```bash
python capture_script.py
```
*   Press **`c`** to capture a snapshot frame and display the analytical confidence metrics directly on the screen.
*   Press **`q`** to safely terminate and release the video capture interface.

---

## 📊 Analytics and Model Benchmarks

The model logic natively processes categorical classifications across seven core emotional dimensions: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**. 

During offline training and evaluation stages against test batches (`7,177` target samples), the network demonstrated high precision metrics on clear-signal classes (e.g., reaching **79% precision on Happy** and **76% precision on Surprise** features), yielding an optimized cross-categorical aggregate test accuracy baseline.

---

## 📦 Core Dependencies (from `requirements.txt`)
*   `flask` - Web infrastructure and routing framework.
*   `deepface` - Face recognition and facial attribute analysis (Emotion, Age, Gender, Race).
*   `opencv-python` - Real-time matrix manipulation and computer vision tracking.
*   `tf-keras` - Deep learning backend optimization plugin.
