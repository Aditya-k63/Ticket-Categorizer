# 🎫 Auto Email / Ticket Categorizer

Classify support tickets into **Billing**, **Technical**, **HR**, or **General** using NLP + Logistic Regression.

Built for the AI/ML Intern Assessment at Fobes Skill Itech.

## Dataset

[**Tobi-Bueck/customer-support-tickets**](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) from Hugging Face — 12,000+ English support tickets across 4 categories.

## Model

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Logistic Regression | 89.0% | 0.700 | 0.897 |
| Naive Bayes | 90.2% | 0.433 | 0.879 |
| Linear SVM | 94.6% | 0.785 | 0.942 |

**Logistic Regression** chosen as primary for its balance of performance + `predict_proba` support (needed for confidence scoring).

## Features

- ✅ Confidence scoring (per-class probabilities)
- ✅ Priority tagging (urgent / normal / low via keyword rules)
- ✅ Human-review threshold at 60% confidence
- ✅ Batch prediction with CSV export
- ✅ Streamlit UI

## Usage

```bash
pip install -r requirements.txt

# Train model
python -m src.train

# Run Streamlit app
streamlit run app.py
```

## Project Structure

```
ticket-categorizer/
├── data/
│   ├── raw/
│   │   ├── fetch.py              # Dataset downloader
│   │   └── tickets.csv
│   └── processed/
├── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py                 # Centralized paths & params
│   ├── preprocess.py             # Text cleaning & lemmatization
│   ├── train.py                  # Model training pipeline
│   ├── evaluate.py               # Metrics (macro F1, per-class)
│   ├── predict.py                # Inference + confidence + priority
│   └── utils.py                  # Save/load helpers
├── app.py                        # Streamlit UI
├── requirements.txt
├── README.md
└── .gitignore
```

## Deployment

### Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Sign in with GitHub → click **New app**
4. Select this repo, branch `main`, file `app.py`
5. Deploy

### Hugging Face Spaces (free)

1. Create a Space at https://huggingface.co/new-space
2. Choose **Streamlit** SDK
3. Push this repo to the Space
4. Set `app.py` as the entry point
