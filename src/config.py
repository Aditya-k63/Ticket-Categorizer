from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "tickets.csv"
MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

CATEGORY_MAP = {
    "Billing and Payments": "Billing",
    "Technical Support": "Technical",
    "IT Support": "Technical",
    "Product Support": "Technical",
    "Service Outages and Maintenance": "Technical",
    "Human Resources": "HR",
    "General Inquiry": "General",
}

CLASSES = ["Billing", "General", "HR", "Technical"]

TFIDF_PARAMS = {
    "lowercase": True,
    "stop_words": "english",
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

LR_PARAMS = {
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": "balanced",
}

HUMAN_REVIEW_THRESHOLD = 0.60
