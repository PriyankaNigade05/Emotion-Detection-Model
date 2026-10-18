from flask import Flask, render_template, request
import os
import cv2
from deepface import DeepFace
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# create uploads folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


# 🔥 Webcam Route
@app.route('/webcam')
def webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']

        cv2.putText(frame, emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Emotion Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or Q
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Webcam closed"


# 🔥 Predict Route
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # unique filename
    filename = str(int(time.time())) + "_" + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(filepath)

    img = cv2.imread(filepath)

    if img is None:
        return "Error loading image"

    result = DeepFace.analyze(
        img,
        actions=['emotion'],
        enforce_detection=False,
        detector_backend='opencv'
    )

    emotion = result[0]['dominant_emotion']

    return render_template('index.html',
                           emotion=emotion,
                           filename=filename)


if __name__ == "__main__":
    app.run(debug=True)