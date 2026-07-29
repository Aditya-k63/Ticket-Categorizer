import numpy as np

from src.config import HUMAN_REVIEW_THRESHOLD, CLASSES
from src.preprocess import clean_text

PRIORITY_KEYWORDS = {
    "high": ["urgent", "down", "not working", "broken", "crash", "error",
             "failed", "outage", "critical", "emergency", "immediately",
             "asap", "blocked", "unable to access", "data loss", "security"],
    "low": ["question", " inquiry", "suggestion", "feature request",
            "would like", "curious", "just wondering", "when", "how to"],
}


def predict_ticket(text: str, model, vectorizer) -> dict:
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    probs = model.predict_proba(vec)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    category = model.classes_[pred_idx]

    text_lower = text.lower()
    priority = "medium"
    for kw in PRIORITY_KEYWORDS["high"]:
        if kw in text_lower:
            priority = "high"
            break
    if priority == "medium":
        for kw in PRIORITY_KEYWORDS["low"]:
            if kw in text_lower:
                priority = "low"
                break

    return {
        "category": category,
        "confidence": round(confidence * 100, 1),
        "priority": priority,
        "needs_review": confidence < HUMAN_REVIEW_THRESHOLD,
        "all_probs": {c: round(float(p) * 100, 1)
                      for c, p in zip(model.classes_, probs)},
    }
