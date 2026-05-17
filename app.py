import os
# ── Suppress TensorFlow / oneDNN noise before any TF import ──────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"   # hide C++ INFO / WARNING / ERROR logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disable oneDNN (removes port.cc msgs)
os.environ["ABSL_MIN_LOG_LEVEL"]    = "3"   # silence absl logging noise
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, render_template, jsonify, request, session
import io, json, random, base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import speech_recognition as sr
from database import engine, SessionLocal
from db_models import Base, SessionResult, VideoData
import cv2
from deepface import DeepFace
import csv
from datetime import datetime
from collections import Counter


# import mysql.connector
# from mysql.connector import Error


from dotenv import load_dotenv
load_dotenv()

from questions import questions
from scripts.predict_lr import predict_with_probs

app = Flask(__name__)
app.secret_key = "supersecretkey"   # required for sessions

# Create tables automatically
Base.metadata.create_all(bind=engine)

# ================= FILE PATHS =================
CSV_FILE = "emotion_log.csv"
IMAGE_FOLDER = "captured_frames"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
RESULTS_CSV = "data/session_results.csv"
os.makedirs("data", exist_ok=True)

# Always reset CSV on startup — clears previous session data
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "timestamp", "angry", "disgust", "fear",
        "happy", "sad", "surprise", "neutral", "dominant"
    ])



# ===== Load Metrics =====
with open("models/tfidf_emotion/metrics.json", "r") as f:
    metrics = json.load(f)
classes = metrics["classes"]

# MySQL insert function
def insert_into_mysql(result):
    try:
        db = SessionLocal()

        new_record = SessionResult(
            patient_id=result["PatientID"],
            anger=result.get("Anger", 0),
            anxiety=result.get("Anxiety", 0),
            depression=result.get("Depression", 0),
            normal_emotion=result.get("Normal", 0),
            personality_disorder=result.get("Personality disorder", 0),
            sadness=result.get("Sadness", 0),
            suicidal=result.get("Suicidal", 0),
        )

        db.add(new_record)
        db.commit()
        db.close()

        return True

    except Exception as e:
        print("SQLAlchemy Error:", e)
        return False


def insert_video_data(patient_id, video_percentages):
    """Save per-session facial emotion percentages to the video_data table."""
    try:
        db = SessionLocal()
        record = VideoData(
            patient_id=patient_id,
            angry=video_percentages.get("angry", 0),
            disgust=video_percentages.get("disgust", 0),
            fear=video_percentages.get("fear", 0),
            happy=video_percentages.get("happy", 0),
            sad=video_percentages.get("sad", 0),
            surprise=video_percentages.get("surprise", 0),
            neutral=video_percentages.get("neutral", 0),
        )
        db.add(record)
        db.commit()
        db.close()
        return True
    except Exception as e:
        print("VideoData DB Error:", e)
        return False

# session["q_index"] = 0

@app.route("/")
def home():
    return render_template("base.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/voice")
def voice():
    return render_template("voice.html")


@app.route("/video")
def video():
    return render_template("video.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/about")
def about():
    return render_template("about.html")



camera_image_saved = False
# ================= VIDEO EMOTION =================
@app.route('/analyze', methods=['POST'])
def analyze():
    global camera_image_saved

    try:
        data = request.json.get('image')

        if not data:
            return jsonify({'error': 'No image data received'})

        encoded_data = data.split(',')[1]
        np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Image decoding failed'})

        result = DeepFace.analyze(
            img,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False
        )

        emotions = {
            key: float(value)
            for key, value in result[0]['emotion'].items()
        }

        dominant_emotion = str(result[0]['dominant_emotion'])

        # SAVE ONLY FIRST CAMERA FRAME
        if not camera_image_saved:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            image_path = os.path.join(
                IMAGE_FOLDER,
                f"camera_access_{timestamp}.jpg"
            )

            cv2.imwrite(image_path, img)

            print(f"Camera access image saved: {image_path}")

            camera_image_saved = True

        # Save to CSV
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


# ================= VOICE SYSTEM =================
@app.route("/start", methods=["POST"])
def start():
    patient_id = request.json.get("patient_id")

    session["patient_id"] = patient_id
    session["q_index"] = 0
    session["responses"] = []
    session["predictions"] = []
    session["probabilities"] = []
    session["questions"] = random.sample(questions, 5)

    return jsonify({"status": "started"})

@app.route("/question")
def get_question():
    q_index = session.get("q_index", 0)
    print(q_index)
    questions_list = session.get("questions", [])

    if q_index >= len(questions_list):
        return jsonify({"done": True})

    current_question = questions_list[q_index]

    # 🔊 Speak question automatically
    subprocess.Popen([
        sys.executable,
        "tts_engine.py",
        current_question
    ])

    return jsonify({
        "question": current_question,
        "index": q_index + 1,
        "total": len(questions_list)
    })


@app.route("/listen", methods=["POST"])
def listen():

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
    except:
        return jsonify({"error": "Sorry, I could not understand."})

    # ---- SAME LOGIC AS /answer ----
    q_index = session.get("q_index", 0)

    if q_index >= len(session["questions"]):
        return jsonify({"done": True})

    pred, probs, _ = predict_with_probs(text)

    session["responses"].append({
        "question": session["questions"][q_index],
        "answer": text
    })

    session["predictions"].append(pred)
    session["probabilities"].append(probs)
    # session["q_index"] += 1

    return jsonify({
        "status": "saved",
        "text": text
    })


@app.route("/answer", methods=["POST"])
def submit_answer():
    answer = request.json.get("answer")
    q_index = session.get("q_index", 0)

    if q_index >= len(session.get("questions", [])):
        return jsonify({"done": True})

    pred, probs, _ = predict_with_probs(answer)

    session["responses"].append({
        "question": session["questions"][q_index-1],
        "answer": answer
    })

    session["predictions"].append(pred)
    session["probabilities"].append(probs)

    # ✅ increment index
    session["q_index"] += 1

    return jsonify({"status": "saved"})


@app.route("/finish")
def finish():
    global camera_image_saved
    camera_image_saved = False

    prob_df = pd.DataFrame(session["probabilities"])
    mean_probs = prob_df.mean().to_dict()
    mean_probs = {cls: round(mean_probs.get(cls, 0) * 100, 2) for cls in classes}

    result = {"PatientID": session["patient_id"]}
    result.update(mean_probs)

    success = insert_into_mysql(result)

    print(session["q_index"])

    # --------- CREATE ATTRACTIVE BAR CHART ----------

    # print(session)
    labels = list(mean_probs.keys())
    values = list(mean_probs.values())

    # Custom colors for each emotion
    emotion_colors = {
        "Anger": "#e74c3c",
        "Anxiety": "#f39c12",
        "Depression": "#8e44ad",
        "Normal": "#2ecc71",
        "Personality disorder": "#3498db",
        "Sadness": "#5dade2",
        "Suicidal": "#2c3e50"
    }

    colors = [emotion_colors.get(label, "#6a11cb") for label in labels]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color=colors)

    plt.ylim(0, 100)
    plt.ylabel("Probability (%)", fontsize=12, fontweight="bold")
    plt.title("Emotion Prediction Result", fontsize=16, fontweight="bold")

    # Rotate labels properly
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)

    # Add percentage values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f'{height:.1f}%',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=200, bbox_inches='tight')
    img.seek(0)
    plt.close()


    chart_base64 = base64.b64encode(img.getvalue()).decode()
    
    
    # ---------------- VIDEO EMOTION ANALYSIS ----------------

    # Read video emotion CSV
    video_df = pd.read_csv("emotion_log.csv")

    # Always show all 7 emotions — compute mean of raw % columns across all frames
    emotion_cols = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    video_percentages = {
        col: round(float(video_df[col].mean()), 2) if col in video_df.columns else 0
        for col in emotion_cols
    }

    # Emotion colors
    video_emotion_colors = {
        "happy": "#2ecc71",
        "sad": "#3498db",
        "angry": "#e74c3c",
        "fear": "#9b59b6",
        "surprise": "#f1c40f",
        "neutral": "#95a5a6",
        "disgust": "#16a085"
    }

    # Labels & values
    video_labels = list(video_percentages.keys())
    video_values = list(video_percentages.values())

    # Matching colors
    video_colors = [
        video_emotion_colors.get(label.lower(), "#6a11cb")
        for label in video_labels
    ]

    # Create chart
    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        video_labels,
        video_values,
        color=video_colors
    )

    plt.ylim(0, 100)

    plt.ylabel(
        "Dominant Emotion Percentage (%)",
        fontsize=12,
        fontweight="bold"
    )

    plt.title(
        "Video Emotion Analysis",
        fontsize=16,
        fontweight="bold"
    )

    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)

    # Add percentage text
    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Convert chart to Base64
    video_img = io.BytesIO()

    plt.savefig(
        video_img,
        format='png',
        dpi=200,
        bbox_inches='tight'
    )

    video_img.seek(0)

    plt.close()

    video_chart_base64 = base64.b64encode(
        video_img.getvalue()
    ).decode()

    # Save video emotion percentages to DB
    insert_video_data(session.get("patient_id", "unknown"), video_percentages)

    return jsonify({
        "result": result,
        "saved": success,
        "audio_chart": chart_base64,
        "video_chart": video_chart_base64,
        "video_percentages": video_percentages
    })


@app.route("/finish_chat")
def finish_chat():
    prob_df = pd.DataFrame(session["probabilities"])
    mean_probs = prob_df.mean().to_dict()
    mean_probs = {cls: round(mean_probs.get(cls, 0) * 100, 2) for cls in classes}

    result = {"PatientID": session["patient_id"]}
    result.update(mean_probs)

    # Save database
    success = insert_into_mysql(result)

    # --------- CREATE ATTRACTIVE BAR CHART ----------

    labels = list(mean_probs.keys())
    values = list(mean_probs.values())

    # Custom colors for each emotion
    emotion_colors = {
        "Anger": "#e74c3c",
        "Anxiety": "#f39c12",
        "Depression": "#8e44ad",
        "Normal": "#2ecc71",
        "Personality disorder": "#3498db",
        "Sadness": "#5dade2",
        "Suicidal": "#2c3e50"
    }

    colors = [emotion_colors.get(label, "#6a11cb") for label in labels]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color=colors)

    plt.ylim(0, 100)
    plt.ylabel("Probability (%)", fontsize=12, fontweight="bold")
    plt.title("Emotion Prediction Result", fontsize=16, fontweight="bold")

    # Rotate labels properly
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)

    # Add percentage values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f'{height:.1f}%',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=200, bbox_inches='tight')
    img.seek(0)
    plt.close()


    chart_base64 = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        "result": result,
        "saved": success,
        "chart": chart_base64
    })


@app.route("/finish_voice")
def finish_voice():
    prob_df = pd.DataFrame(session["probabilities"])
    mean_probs = prob_df.mean().to_dict()
    mean_probs = {cls: round(mean_probs.get(cls, 0) * 100, 2) for cls in classes}

    result = {"PatientID": session["patient_id"]}
    result.update(mean_probs)

    success = insert_into_mysql(result)

    print(session["q_index"])

    # --------- CREATE ATTRACTIVE BAR CHART ----------

    # print(session)
    labels = list(mean_probs.keys())
    values = list(mean_probs.values())

    # Custom colors for each emotion
    emotion_colors = {
        "Anger": "#e74c3c",
        "Anxiety": "#f39c12",
        "Depression": "#8e44ad",
        "Normal": "#2ecc71",
        "Personality disorder": "#3498db",
        "Sadness": "#5dade2",
        "Suicidal": "#2c3e50"
    }

    colors = [emotion_colors.get(label, "#6a11cb") for label in labels]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color=colors)

    plt.ylim(0, 100)
    plt.ylabel("Probability (%)", fontsize=12, fontweight="bold")
    plt.title("Emotion Prediction Result", fontsize=16, fontweight="bold")

    # Rotate labels properly
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)

    # Add percentage values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f'{height:.1f}%',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=200, bbox_inches='tight')
    img.seek(0)
    plt.close()


    chart_base64 = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        "result": result,
        "saved": success,
        "chart": chart_base64
    })

# @app.route("/start_chat", methods=["POST"])
# def start_chat():
#     patient_id = request.json.get("patient_id")

#     session["patient_id"] = patient_id
#     session["q_index"] = 0
#     session["responses"] = []
#     session["predictions"] = []
#     session["probabilities"] = []
#     session["questions"] = random.sample(questions, 5)

#     return jsonify({"status": "started"})

@app.route("/question_chat")
def get_question_chat():
    q_index = session.get("q_index", 0)
    qs = session.get("questions", [])
    questions_list = session.get("questions", [])

    if q_index >= len(questions_list):
        return jsonify({"done": True})

    if q_index < len(qs):
        return jsonify({
            "question": qs[q_index],
            "index": q_index + 1,
            "total": len(qs)
        })
    else:
        return jsonify({"done": True})

# @app.route("/answer_chat", methods=["POST"])
# def submit_answer_chat():
#     answer = request.json.get("answer")
#     q_index = session["q_index"]

#     pred, probs, _ = predict_with_probs(answer)

#     session["responses"].append({
#         "question": session["questions"][q_index],
#         "answer": answer
#     })

#     session["predictions"].append(pred)
#     session["probabilities"].append(probs)
#     session["q_index"] += 1

#     return jsonify({"status": "saved"})


# @app.route("/finish_chat")
# def finish_chat():
#     prob_df = pd.DataFrame(session["probabilities"])
#     mean_probs = prob_df.mean().to_dict()
#     mean_probs = {cls: round(mean_probs.get(cls, 0) * 100, 2) for cls in classes}

#     result = {"PatientID": session["patient_id"]}
#     result.update(mean_probs)

#     # Save database
#     success = insert_into_mysql(result)

#     # --------- CREATE ATTRACTIVE BAR CHART ----------

#     labels = list(mean_probs.keys())
#     values = list(mean_probs.values())

#     # Custom colors for each emotion
#     emotion_colors = {
#         "Anger": "#e74c3c",
#         "Anxiety": "#f39c12",
#         "Depression": "#8e44ad",
#         "Normal": "#2ecc71",
#         "Personality disorder": "#3498db",
#         "Sadness": "#5dade2",
#         "Suicidal": "#2c3e50"
#     }

#     colors = [emotion_colors.get(label, "#6a11cb") for label in labels]

#     plt.figure(figsize=(12, 6))
#     bars = plt.bar(labels, values, color=colors)

#     plt.ylim(0, 100)
#     plt.ylabel("Probability (%)", fontsize=12, fontweight="bold")
#     plt.title("Emotion Prediction Result", fontsize=16, fontweight="bold")

#     # Rotate labels properly
#     plt.xticks(rotation=30, ha='right', fontsize=11)
#     plt.yticks(fontsize=11)

#     # Add percentage values on top of bars
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,
#             height + 2,
#             f'{height:.1f}%',
#             ha='center',
#             fontsize=10,
#             fontweight='bold'
#         )

#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.tight_layout()

#     img = io.BytesIO()
#     plt.savefig(img, format='png', dpi=200, bbox_inches='tight')
#     img.seek(0)
#     plt.close()


#     chart_base64 = base64.b64encode(img.getvalue()).decode()

#     return jsonify({
#         "result": result,
#         "saved": success,
#         "chart": chart_base64
#     })

    

if __name__ == "__main__":
    app.run(debug=True)