from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from deepface import DeepFace
import csv
from datetime import datetime
import os

app = Flask(__name__)

CSV_FILE = "emotion_log.csv"

# Create CSV file with header if not exists
if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
            "dominant"
        ])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json.get('image')

        if not data:
            return jsonify({'error': 'No image data received'})

        # Remove base64 header
        encoded_data = data.split(',')[1]
        np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Image decoding failed'})

        # Save debug frame to check what server receives
        cv2.imwrite("debug_frame.jpg", img)

        # Run DeepFace emotion detection
        result = DeepFace.analyze(
            img,
            actions=['emotion'],
            detector_backend='opencv',   # You can try 'retinaface' if needed
            enforce_detection=False
        )

        emotions = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']

        # Save results to CSV
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now(),
                emotions.get('angry', 0),
                emotions.get('disgust', 0),
                emotions.get('fear', 0),
                emotions.get('happy', 0),
                emotions.get('sad', 0),
                emotions.get('surprise', 0),
                emotions.get('neutral', 0),
                dominant_emotion
            ])

        return jsonify({
            'dominant_emotion': dominant_emotion,
            'emotions': emotions
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)