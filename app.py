import streamlit as st
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODEL_PATH, VECTORIZER_PATH
from src.utils import load_model, load_vectorizer
from src.predict import predict_ticket
from src.train import train


@st.cache_resource
def load_models():
    if not MODEL_PATH.exists():
        st.info("Training model for the first time...")
        train()
    m = load_model(MODEL_PATH)
    v = load_vectorizer(VECTORIZER_PATH)
    return m, v


st.set_page_config(page_title="Ticket Categorizer", page_icon="🎫", layout="centered")
st.title("🎫 Ticket Categorizer")
st.caption("Type a support ticket below — it classifies automatically.")

model, vectorizer = load_models()

ticket = st.text_area(
    "Ticket text",
    placeholder="e.g. I was charged twice for my subscription this month.",
    height=100,
    label_visibility="collapsed",
)

if ticket.strip():
    result = predict_ticket(ticket.strip(), model, vectorizer)

    if result["needs_review"]:
        st.warning(f"⚠ Needs Human Review  ·  Priority: {result['priority'].title()}")
    else:
        st.success(f"✅ {result['category']}  ·  {result['confidence']}% confidence  ·  Priority: {result['priority'].title()}")

    st.progress(result["confidence"] / 100)

    cols = st.columns(4)
    for i, (cat, prob) in enumerate(result["all_probs"].items()):
        highlight = "background:#2C5FDB;color:white;border-radius:8px;padding:8px;text-align:center;" if cat == result["category"] else "background:#EEF1F6;border-radius:8px;padding:8px;text-align:center;"
        cols[i].markdown(
            f"<div style='{highlight}'><div style='font-size:12px'>{cat}</div><div style='font-size:22px;font-weight:700'>{prob:.0f}%</div></div>",
            unsafe_allow_html=True,
        )
else:
    st.info("Enter a ticket above to see the prediction.")
