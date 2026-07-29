import pandas as pd
import numpy as np
import re
import nltk
import joblib

from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

CATEGORY_MAP = {
    "Billing and Payments": "Billing",
    "Technical Support": "Technical",
    "IT Support": "Technical",
    "Product Support": "Technical",
    "Service Outages and Maintenance": "Technical",
    "Human Resources": "HR",
    "General Inquiry": "General",
}

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


def predict_ticket(ticket, model, vectorizer):
    clean = preprocess(ticket)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return {"category": pred, "confidence": round(prob, 3), "needs_review": prob < 0.60}


def evaluate(model, name, X_test_vec, y_test, classes):
    pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, pred)

    report = classification_report(y_test, pred, digits=3)
    cm = confusion_matrix(y_test, pred)

    per_class_f1 = f1_score(y_test, pred, average=None)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")

    per_class_precision = precision_score(y_test, pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, pred, average=None, zero_division=0)

    print("\n" + "=" * 55)
    print(f"  {name}")
    print("=" * 55)
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Weighted F1   : {weighted_f1:.4f}")

    try:
        proba = model.predict_proba(X_test_vec)
        ll = log_loss(y_test, proba)
        print(f"  Log Loss      : {ll:.4f}")
    except Exception:
        print(f"  Log Loss      : N/A (no predict_proba)")

    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<12} {'Prec':>7} {'Recall':>7} {'F1':>7}  {'Support':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7}  {'-'*8}")
    for i, cls in enumerate(classes):
        support = (y_test == cls).sum()
        print(f"  {cls:<12} {per_class_precision[i]:>7.3f} {per_class_recall[i]:>7.3f} "
              f"{per_class_f1[i]:>7.3f}  {support:>8}")

    print(f"\n  Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in classes],
                         columns=[f"Pred {c}" for c in classes])
    for line in cm_df.to_string().split("\n"):
        print(f"    {line}")

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "per_class_f1": dict(zip(classes, per_class_f1)),
            "per_class_precision": dict(zip(classes, per_class_precision)),
            "per_class_recall": dict(zip(classes, per_class_recall))}


def print_comparison_table(results, classes):
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON")
    print("=" * 65)
    header = f"  {'Metric':<20}"
    for name in results:
        header += f" {name:>12}"
    print(header)
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

    row = f"  {'Accuracy':<20}"
    for r in results.values():
        row += f" {r['accuracy']:>11.3f} "
    print(row)

    row = f"  {'Macro F1':<20}"
    for r in results.values():
        row += f" {r['macro_f1']:>11.3f} "
    print(row)

    row = f"  {'Weighted F1':<20}"
    for r in results.values():
        row += f" {r['weighted_f1']:>11.3f} "
    print(row)

    print(f"\n  Per-class F1:")
    f1_header = f"  {'Class':<12}"
    for name in results:
        f1_header += f" {name:>12}"
    print(f1_header)
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for cls in classes:
        row = f"  {cls:<12}"
        for r in results.values():
            row += f" {r['per_class_f1'][cls]:>11.3f} "
        print(row)


def main():
    print("Loading dataset...")
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    df = pd.DataFrame(ds)

    df = df[df["language"] == "en"].copy()
    print(f"Total English tickets: {len(df)}")

    df = df[df["queue"].isin(CATEGORY_MAP.keys())].copy()
    df["category"] = df["queue"].map(CATEGORY_MAP)
    classes = sorted(df["category"].unique())
    print(f"After category filter: {len(df)}")
    print(df["category"].value_counts().to_string())

    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df = df[["text", "category"]]

    print("Preprocessing text...")
    df["clean_text"] = df["text"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"],
        test_size=0.2, random_state=42, stratify=df["category"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", max_features=10000,
        ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_model.fit(X_train_vec, y_train)

    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_train_vec, y_train)

    results = {}
    results["NB"] = evaluate(nb_model, "Naive Bayes", X_test_vec, y_test, classes)
    results["LR"] = evaluate(lr_model, "Logistic Regression", X_test_vec, y_test, classes)
    results["SVM"] = evaluate(svm_model, "Linear SVM", X_test_vec, y_test, classes)

    print_comparison_table(results, classes)

    joblib.dump(lr_model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("\nModel saved to models/")

    samples = [
        "I paid yesterday but still haven't received my refund.",
        "My laptop crashes whenever I open the application.",
        "I want to apply for leave next week.",
        "Where is your company located?",
    ]
    print("\n" + "=" * 55)
    print("  Sample Predictions (Logistic Regression)")
    print("=" * 55)
    for ticket in samples:
        result = predict_ticket(ticket, lr_model, vectorizer)
        flag = " !! REVIEW" if result["needs_review"] else ""
        print(f"\n  Ticket: {ticket}")
        print(f"  -> {result['category']} (conf: {result['confidence']:.1%}){flag}")
        print(f"     Needs review: {result['needs_review']}")


if __name__ == "__main__":
    main()
