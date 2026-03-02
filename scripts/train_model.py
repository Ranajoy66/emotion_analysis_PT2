# scripts/train_model_lightweight.py
import os
import json
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from scripts.preprocessing import clean_text

# ===== Paths =====
DATA_PATH = "data/fast_data_augmented_1L_each.csv"
MODEL_DIR = "models/tfidf_emotion"
os.makedirs(MODEL_DIR, exist_ok=True)

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
LABEL2ID_PATH = os.path.join(MODEL_DIR, "label2id.json")

# ===== Load dataset =====
df = pd.read_csv(DATA_PATH, encoding="latin1")
df.columns = df.columns.str.strip()

df["clean_text"] = df["Text"].astype(str).apply(clean_text)

MAX_SAMPLES_PER_CLASS = 400

balanced_df = (
    df
    .groupby("Label", as_index=False)
    .sample(n=MAX_SAMPLES_PER_CLASS, random_state=42)
)

texts = balanced_df["clean_text"]
labels = balanced_df["Label"]

# ===== Label encoding =====
unique_labels = sorted(labels.unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
labels_id = labels.map(label2id)

# ===== Train-test split =====
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels_id,
    test_size=0.2,
    random_state=42,
    stratify=labels_id
)

# ===== TF-IDF Vectorization =====
print("⏳ Training TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ===== SMOTE =====
print("⚖️ Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=2)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ===== Class weights =====
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_bal),
    y=y_train_bal
)
class_weight_dict = dict(enumerate(class_weights))

# ===== Train Logistic Regression =====
print("🚀 Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    class_weight=class_weight_dict,
    n_jobs=-1
)
lr.fit(X_train_bal, y_train_bal)

# ===== Evaluate =====
y_pred = lr.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=unique_labels,
    digits=4
)

print(f"✅ Accuracy: {acc * 100:.2f}%")
print(report)

# ===== Save =====
joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)
joblib.dump(lr, LR_PATH, compress=3)

with open(METRICS_PATH, "w") as f:
    json.dump(
        {"accuracy": acc, "report": report, "classes": unique_labels},
        f,
        indent=4
    )

with open(LABEL2ID_PATH, "w") as f:
    json.dump(label2id, f, indent=4)

print(f"✅ TF-IDF model saved to {MODEL_DIR}")
