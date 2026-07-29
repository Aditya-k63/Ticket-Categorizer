import joblib
import numpy as np

from .preprocessing import clean_text

PRIORITY_KEYWORDS = {
    "high": ["urgent", "down", "not working", "broken", "crash", "error",
             "failed", "outage", "critical", "emergency", "immediately",
             "asap", "blocked", "unable to access", "data loss", "security"],
    "low": ["question", " inquiry", "suggestion", "feature request",
            "would like", "curious", "just wondering", "when", "how to"],
}

HUMAN_REVIEW_THRESHOLD = 0.60


class TicketClassifier:
    def __init__(self, model_dir: str = "models"):
        self.vectorizer = joblib.load(f"{model_dir}/vectorizer.joblib")
        self.clf = joblib.load(f"{model_dir}/classifier.joblib")
        self.classes = joblib.load(f"{model_dir}/classes.joblib")

    def predict(self, text: str) -> dict:
        clean = clean_text(text)
        vec = self.vectorizer.transform([clean])
        probs = self.clf.predict_proba(vec)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        category = self.classes[pred_idx]

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

        confidence_pct = round(confidence * 100, 1)

        return {
            "text": text,
            "category": category,
            "confidence": confidence_pct,
            "priority": priority,
            "needs_review": confidence < HUMAN_REVIEW_THRESHOLD,
            "all_classes": list(self.classes),
            "all_probs": [round(float(p) * 100, 1) for p in probs],
        }


def predict_ticket(text: str, model_dir: str = "models") -> dict:
    clf = TicketClassifier(model_dir)
    return clf.predict(text)


if __name__ == "__main__":
    sample = "My internet is down and I can't work, please fix it urgently!"
    result = predict_ticket(sample)
    for k, v in result.items():
        print(f"{k}: {v}")
