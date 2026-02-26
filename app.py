from flask import Flask, render_template, jsonify, request, session
import os, io, json, random, base64
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import sys
import speech_recognition as sr
# import mysql.connector
# from mysql.connector import Error
from database import engine, SessionLocal
from db_models import Base, SessionResult


from dotenv import load_dotenv
load_dotenv()

from questions import questions
from scripts.predict_lr import predict_with_probs

app = Flask(__name__)
app.secret_key = "supersecretkey"   # required for sessions

# Create tables automatically
Base.metadata.create_all(bind=engine)


# ===== Paths =====
RESULTS_CSV = "data/session_results.csv"
os.makedirs("data", exist_ok=True)

# ===== Load Metrics =====
with open("models/tfidf_emotion/metrics.json", "r") as f:
    metrics = json.load(f)
classes = metrics["classes"]


# # MySQL connectionProvider
# def get_db_connection():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password=os.environ.get("MYSQL_PASSWORD"),
#         database="emotion_analysis"
#     )


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

# session["q_index"] = 0

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/survey")
def index():
    return render_template("index.html")

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


# @app.route("/answer", methods=["POST"])
# def submit_answer():
#     answer = request.json.get("answer")
#     q_index = session["q_index"]

#     pred, probs, _ = predict_with_probs(answer)

#     session["responses"].append({
#         "question": session["questions"][q_index],
#         "answer": answer
#     })

#     session["predictions"].append(pred)
#     session["probabilities"].append(probs)
#     # session["q_index"] += 1

#     return jsonify({"status": "saved"})

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

    

if __name__ == "__main__":
    app.run(debug=True)