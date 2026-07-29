import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
import joblib
import os

from .preprocessing import clean_text


def train(data_path: str = "data/tickets.csv",
          model_dir: str = "models",
          use_logreg: bool = False):
    df = pd.read_csv(data_path)
    df["clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if use_logreg:
        clf = LogisticRegression(max_iter=500, multi_class="multinomial", random_state=42)
        model_name = "logistic_regression"
    else:
        clf = MultinomialNB(alpha=0.1)
        model_name = "naive_bayes"

    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm,
                       index=sorted(clf.classes_),
                       columns=sorted(clf.classes_)))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vectorizer, f"{model_dir}/vectorizer.joblib")
    joblib.dump(clf, f"{model_dir}/classifier.joblib")
    joblib.dump(list(clf.classes_), f"{model_dir}/classes.joblib")
    print(f"\nModel saved to {model_dir}/")

    return vectorizer, clf


if __name__ == "__main__":
    train()
