# 🎫 Auto Email / Ticket Categorizer

Classify support tickets into **Billing**, **Technical**, **HR**, or **General** using NLP and Logistic Regression. Built for the AI/ML Intern Assessment at **Fobes Skill Itech**.

👉 **Live demo**: [Streamlit Cloud](https://share.streamlit.io) (deploy from this repo)

---

## Dataset

[**Tobi-Bueck/customer-support-tickets**](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) — 61,765 multilingual support tickets from Hugging Face. Filtered to **~12,000 English tickets** across 4 categories:

| Category   | Samples |
|------------|---------|
| Technical  | 8,149   |
| Billing    | 2,897   |
| HR         | 553     |
| General    | 404     |

The dataset is imbalanced (Technical dominates). Raw data is fetched via `data/raw/fetch.py`.

---

## Model

### Training pipeline

| Step | Detail |
|------|--------|
| **Preprocessing** | Lowercase, remove URLs/emails/punctuation, keep digits, stopword removal, lemmatization |
| **Vectorization** | TF-IDF (10K features, unigrams + bigrams, sublinear TF) |
| **Classifier** | Logistic Regression (`class_weight="balanced"`) |
| **Train/Test split** | 80/20, stratified |

### Model comparison

| Metric              | Naive Bayes | Logistic Regression | Linear SVM |
|---------------------|-------------|-------------------|------------|
| Accuracy            | 90.2%       | 89.0%             | **94.6%**  |
| Macro F1            | 0.433       | 0.700             | **0.785**  |
| Weighted F1         | 0.879       | 0.897             | **0.942**  |

### Why Logistic Regression was deployed

Linear SVM has the highest accuracy (94.6%) but **no `predict_proba`** — it can't output confidence scores. Since confidence-based human-review routing is a core feature, **Logistic Regression** was chosen as the primary model. It provides calibrated probabilities with competitive performance.

### Per-class F1 (Logistic Regression)

| Class    | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| Billing  | 0.721     | 0.877  | 0.791 | 579     |
| General  | 0.434     | 0.728  | 0.544 | 81      |
| HR       | 0.450     | 0.649  | 0.531 | 111     |
| Technical| 0.966     | 0.904  | 0.934 | 3579    |

---

## Features

### Confidence scoring

Returns the probability distribution across all 4 classes. The top prediction's confidence determines auto-routing vs. human review.

### Human-review logic

A ticket is flagged for review when **any** condition is true:

1. **Top confidence < 50%** — the model isn't certain enough
2. **Top-2 margin < 15%** — the model is torn between two classes
3. **< 4 meaningful words** — the ticket is too short/vague to classify (e.g. "Help", "Need assistance")

### Priority tagging

Keyword-based priority assignment:

| Priority | Trigger words |
|----------|---------------|
| High     | urgent, down, crash, error, failed, outage, critical, emergency, blocked, data loss, security |
| Low      | question, inquiry, suggestion, feature request, curious, just wondering |
| Medium   | (default) |

---

## Known limitations

1. **TF-IDF lacks context.** Words like "account" appear in both billing tickets ("account balance") and technical tickets ("account access"). When a ticket says "I forgot my password and cannot log in to my account," the model may overweigh "account" and misclassify as Billing. A contextual model like BERT would handle this better.

2. **Single-label constraint.** A ticket mentioning "payment failed" and "cannot log in" (two issues) is forced into one category. A production system might multi-label or split into sub-tickets.

3. **Imbalanced dataset.** HR and General have few samples (553 and 404 vs 8,149 Technical). `class_weight="balanced"` helps, but minority classes still underperform.

4. **Short/vague queries.** Tickets like "I need help" contain insufficient signal. The <4-word heuristic catches these, but a production system would also use conversation history or user metadata.

---

## Usage

```bash
pip install -r requirements.txt

# Train model
python -m src.train

# Launch Streamlit app
streamlit run app.py
```

**Controls**: Press **Enter** to classify · **Shift+Enter** for new line

---

## Project structure

```
ticket-categorizer/
├── data/
│   └── raw/
│       ├── fetch.py              # Download from Hugging Face
│       └── tickets.csv           # Labeled dataset
├── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── .gitkeep
├── src/
│   ├── config.py                 # Paths, params, constants
│   ├── preprocess.py             # Cleaning + lemmatization
│   ├── train.py                  # TF-IDF + Logistic Regression
│   ├── evaluate.py               # Macro F1, per-class metrics
│   ├── predict.py                # Inference, confidence, priority
│   └── utils.py                  # Save/load helpers
├── app.py                        # Streamlit UI
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Deployment

### Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → **New app**
4. Select repo, branch `master`, file `app.py`
5. **Deploy**

The pre-trained model is included in the repo, so no training step runs on the cloud.

---

## What would I improve with more data or time?

- **Multi-label classification** to handle compound tickets
- **Cross-validation + GridSearchCV** for hyperparameter tuning
- **BERT or DistilBERT** for contextual understanding (especially for ambiguous terms like "account")
- **Active learning** to surface low-confidence tickets for manual labeling and retraining
- **Docker + CI/CD** for reproducible deployment
- **A/B testing framework** to measure whether auto-routing improves resolution time
