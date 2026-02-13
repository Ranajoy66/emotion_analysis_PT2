# scripts/session_logger.py
import os
import pandas as pd
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def build_columns(classes):
    # base columns
    cols = [
        "Person_ID",
        "Session_Timestamp",
        "Input_Text",
        "Predicted_Emotion",
        "Predicted_Emotion_Pct",
    ]
    # per-class percent columns
    for c in classes:
        safe = c.replace(" ", "_").replace("-", "_")
        cols.append(f"Emotion_{safe}_Pct")
    cols.append("Session_No")
    return cols

def next_session_no(csv_path, person_id):
    if not os.path.exists(csv_path):
        return 1
    try:
        df = pd.read_csv(csv_path)
        if "Person_ID" not in df.columns or "Session_No" not in df.columns:
            return 1
        prev = df.loc[df["Person_ID"] == person_id, "Session_No"]
        return int(prev.max()) + 1 if not prev.empty else 1
    except Exception:
        return 1

def append_session(csv_path, person_id, input_text, pred_label, probs_dict, classes, ts=None):
    ensure_dir(csv_path)
    if ts is None:
        ts = datetime.now().isoformat(timespec="seconds")  # store local time; use pytz if you need IST

    cols = build_columns(classes)
    # create row dict with all columns present
    row = {c: None for c in cols}
    row["Person_ID"] = person_id
    row["Session_Timestamp"] = ts
    row["Input_Text"] = input_text
    row["Predicted_Emotion"] = pred_label
    row["Predicted_Emotion_Pct"] = round(probs_dict[pred_label] * 100, 2)

    # fill per-class percentages
    for c in classes:
        safe = c.replace(" ", "_").replace("-", "_")
        row[f"Emotion_{safe}_Pct"] = round(probs_dict[c] * 100, 2)

    # session number per patient
    row["Session_No"] = next_session_no(csv_path, person_id)

    # write/append
    write_header = not os.path.exists(csv_path)
    df_row = pd.DataFrame([row], columns=cols)
    df_row.to_csv(csv_path, mode="a", index=False, header=write_header)
