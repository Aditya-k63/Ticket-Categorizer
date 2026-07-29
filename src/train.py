from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import TFIDF_PARAMS, LR_PARAMS, MODEL_PATH, VECTORIZER_PATH
from src.preprocess import load_raw_data, preprocess_dataframe
from src.utils import save_model, save_vectorizer


def train():
    df = load_raw_data()
    df = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"],
        test_size=0.2, random_state=42, stratify=df["category"],
    )

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train_vec, y_train)

    save_model(model, MODEL_PATH)
    save_vectorizer(vectorizer, VECTORIZER_PATH)

    return model, vectorizer, X_test, y_test


if __name__ == "__main__":
    train()
    print("Model trained and saved.")
