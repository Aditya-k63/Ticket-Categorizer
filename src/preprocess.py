import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

from src.config import CATEGORY_MAP, RAW_DATA_PATH

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "text" in df.columns and "category" in df.columns:
        return df[["text", "category"]]

    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df["category"] = df["queue"].map(CATEGORY_MAP)
    df = df[df["category"].notna()].copy()
    df = df[["text", "category"]]
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    return df
