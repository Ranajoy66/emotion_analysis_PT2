import re
import pandas as pd

def clean_text(text) -> str:
    """Basic text cleaning with safe type handling"""
    if not isinstance(text, str):  
        text = str(text) if text is not None else ""  # convert non-strings / NaN to empty string
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

