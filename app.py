import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODEL_PATH, VECTORIZER_PATH, CLASSES, HUMAN_REVIEW_THRESHOLD
from src.utils import load_model, load_vectorizer
from src.predict import predict_ticket
from src.evaluate import evaluate_model
from src.train import train
from src.preprocess import load_raw_data, preprocess_dataframe
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="Ticket Categorizer",
    page_icon="🎫",
    layout="centered",
)

st.title("🎫 Auto Email / Ticket Categorizer")
st.markdown("Classify support tickets into **Billing**, **Technical**, **HR**, or **General** with confidence scoring & priority tagging.")

# ---------- load model ----------
@st.cache_resource
def load_models():
    if not MODEL_PATH.exists():
        st.info("No trained model found. Training now...")
        train()
    model = load_model(MODEL_PATH)
    vectorizer = load_vectorizer(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_models()

# ---------- tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Evaluate", "📦 Batch Predict", "ℹ️ About"])

# ========= TAB 1: Predict =========
with tab1:
    st.subheader("Classify a single ticket")
    ticket_text = st.text_area(
        "Enter ticket text",
        placeholder="e.g. I was charged twice for my subscription this month.",
        height=120,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("Classify", type="primary", use_container_width=True)

    if predict_btn and ticket_text.strip():
        with st.spinner("Classifying..."):
            result = predict_ticket(ticket_text.strip(), model, vectorizer)

        # result card
        if result["needs_review"]:
            st.warning("⚠️ Needs Human Review (confidence below threshold)")
        else:
            st.success("✅ Auto-classified")

        c1, c2, c3 = st.columns(3)
        c1.metric("Category", result["category"])
        c2.metric("Confidence", f"{result['confidence']:.1f}%")
        c3.metric("Priority", result["priority"].title())

        st.markdown("**Per-class probabilities:**")
        prob_df = pd.DataFrame([
            {"Category": cat, "Probability": f"{prob:.1f}%"}
            for cat, prob in result["all_probs"].items()
        ])
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

        # horizontal bar chart
        fig, ax = plt.subplots(figsize=(6, 2.5))
        cats = list(result["all_probs"].keys())
        vals = list(result["all_probs"].values())
        colors = ["#2C5FDB" if c == result["category"] else "#DCE2ED" for c in cats]
        ax.barh(cats, vals, color=colors)
        ax.axvline(HUMAN_REVIEW_THRESHOLD * 100, color="red", ls="--", label=f"Review threshold ({HUMAN_REVIEW_THRESHOLD*100:.0f}%)")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    elif predict_btn:
        st.warning("Please enter ticket text.")

# ========= TAB 2: Evaluate =========
with tab2:
    st.subheader("Model Performance")

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Evaluating..."):
            df = load_raw_data()
            df = preprocess_dataframe(df)
            _, X_test, _, y_test = train_test_split(
                df["clean_text"], df["category"],
                test_size=0.2, random_state=42, stratify=df["category"],
            )

            if not hasattr(model, "classes_"):
                model.classes_ = CLASSES

            custom_vectorizer = load_vectorizer(VECTORIZER_PATH)
            X_test_vec = custom_vectorizer.transform(X_test)

            pred = model.predict(X_test_vec)

            from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

            acc = accuracy_score(y_test, pred)
            macro_f1 = f1_score(y_test, pred, average="macro")
            weighted_f1 = f1_score(y_test, pred, average="weighted")

            st.metric("Accuracy", f"{acc:.2%}")
            col_a, col_b = st.columns(2)
            col_a.metric("Macro F1", f"{macro_f1:.2%}")
            col_b.metric("Weighted F1", f"{weighted_f1:.2%}")

            st.markdown("**Classification Report:**")
            st.text(classification_report(y_test, pred, digits=3))

            st.markdown("**Confusion Matrix:**")
            cm = confusion_matrix(y_test, pred, labels=CLASSES)
            cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
            st.dataframe(cm_df, use_container_width=True)

# ========= TAB 3: Batch Predict =========
with tab3:
    st.subheader("Batch classify multiple tickets")
    st.markdown("Paste one ticket per line.")
    batch_text = st.text_area(
        "Batch input",
        placeholder="Ticket 1 text...\nTicket 2 text...\nTicket 3 text...",
        height=150,
    )

    if st.button("Classify All", type="primary") and batch_text.strip():
        lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
        results = []
        for ticket in lines:
            r = predict_ticket(ticket, model, vectorizer)
            results.append({
                "Ticket": ticket[:60] + ("..." if len(ticket) > 60 else ""),
                "Category": r["category"],
                "Confidence": f"{r['confidence']:.1f}%",
                "Priority": r["priority"].title(),
                "Review": "⚠️ Yes" if r["needs_review"] else "✅ No",
            })
        st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
        csv = pd.DataFrame(results).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# ========= TAB 4: About =========
with tab4:
    st.subheader("About this project")
    st.markdown("""
    **Auto Email / Ticket Categorizer** — AI/ML Intern Assessment for Fobes Skill Itech.

    - **Dataset**: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (12K+ English tickets)
    - **Model**: Logistic Regression with TF-IDF vectorization
    - **Features**:
      - Per-class confidence scores
      - Priority tagging (urgent / normal / low)
      - Human-review threshold at 60% confidence
      - Batch prediction with CSV export
    """)
