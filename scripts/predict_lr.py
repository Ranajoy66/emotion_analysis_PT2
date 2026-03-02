import os
import joblib
import json
import numpy as np
import requests

# ===== Config =====
HF_REPO = "Sutanu-59/emotion-analysis-models"
MODEL_DIR = "models/tfidf_emotion"

FILES = [
    "tfidf_vectorizer.pkl",
    "logistic_regression.pkl",
    "label2id.json"
]

os.makedirs(MODEL_DIR, exist_ok=True)

# def download_from_hf_if_needed():
#     base_url = f"https://huggingface.co/{HF_REPO}/resolve/main/"
#     for fname in FILES:
#         fpath = os.path.join(MODEL_DIR, fname)
#         if not os.path.exists(fpath):
#             print(f"⬇️ Downloading {fname}...")
#             r = requests.get(base_url + fname)
#             if r.status_code != 200:
#                 raise RuntimeError(f"Failed to download {fname}")
#             with open(fpath, "wb") as f:
#                 f.write(r.content)

# ===== Load models =====
print("⏳ Loading TF-IDF emotion model...")
# download_from_hf_if_needed()

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")
LABEL2ID_PATH = os.path.join(MODEL_DIR, "label2id.json")

vectorizer = joblib.load(VECTORIZER_PATH)
lr_model = joblib.load(LR_PATH)

with open(LABEL2ID_PATH, "r") as f:
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}
CLASSES = [id2label[i] for i in range(len(id2label))]

# ===== Prediction =====
def predict_with_probs(text: str):
    X = vectorizer.transform([text])
    proba = lr_model.predict_proba(X)[0]

    pred_idx = int(np.argmax(proba))
    pred_label = id2label[pred_idx]

    probs_dict = {
        id2label[i]: float(p)
        for i, p in enumerate(proba)
    }

    return pred_label, probs_dict, CLASSES
