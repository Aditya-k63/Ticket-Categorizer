from pathlib import Path
from datasets import load_dataset
import pandas as pd
import re

QUEUE_MAP = {
    "Technical Support": "Technical",
    "Billing and Payments": "Billing",
    "Human Resources": "HR",
    "General Inquiry": "General",
}


def clean_text(text: str) -> str:
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_and_save(output_path: str | None = None):
    if output_path is None:
        output_path = str(Path(__file__).resolve().parent / "tickets.csv")

    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")

    rows = []
    for row in ds:
        queue = row["queue"]
        cat = QUEUE_MAP.get(queue)
        if cat is None or row["language"] != "en":
            continue

        subject = clean_text(row["subject"] or "")
        body = clean_text(row["body"] or "")
        text = f"{subject}. {body}" if subject else body

        rows.append({
            "text": text,
            "category": cat,
            "priority": row.get("priority", "medium"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} English tickets to {output_path}")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    fetch_and_save()
