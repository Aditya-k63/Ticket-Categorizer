import re
import string

STOPWORDS = set(
    "a an the and or but in on at to for of with by from is are was were be "
    "been being have has had do does did will would could should may might "
    "shall can need dare ought used must i you he she it we they my me "
    "your his her its our their this that these those am no not nor so "
    "up down out off over under again further then once here there "
    "all each every both few more most other some such only own same "
    "too very just about also much many".split()
)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)
