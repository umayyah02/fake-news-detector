# src/preprocess.py

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download once — safe to call repeatedly, nltk skips if already present
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]")


def clean_text(text: str) -> str:
    """
    Single source of truth for text cleaning.
    Used by both train.py and predict.py to avoid train/inference skew.

    Steps: lowercase -> strip HTML/URLs -> remove punctuation ->
    remove stopwords.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = NON_ALPHANUMERIC_PATTERN.sub(" ", text)

    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]

    return " ".join(tokens).strip()


def build_processed_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw merged DataFrame (from data_loader.load_raw_data)
    and produces the processed schema: content, label, source_dataset.

    content = cleaned(title + " " + text)
    """
    df = raw_df.copy()

    combined = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content"] = combined.apply(clean_text)

    processed = df[["content", "label", "source_dataset"]].copy()

    # Drop rows that became empty after cleaning (e.g. bad/junk rows)
    before = len(processed)
    processed = processed[processed["content"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(processed)
    if dropped > 0:
        print(f"Warning: dropped {dropped} rows with empty content after cleaning")

    return processed


if __name__ == "__main__":
    # Quick manual smoke test
    sample = "Check THIS out!! <b>Breaking</b> news at http://fake.com — unbelievable??"
    print("Before:", sample)
    print("After: ", clean_text(sample))

    from data_loader import load_raw_data

    raw_df = load_raw_data("data/raw/Fake.csv", "data/raw/True.csv")
    processed_df = build_processed_dataset(raw_df)
    print(processed_df.shape)
    print(processed_df.head())