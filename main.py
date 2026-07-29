#!/usr/bin/env python3
"""
Auto Email / Ticket Categorizer
Fobes Skill Itech — AI/ML Intern Assessment
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data.fetch_dataset import fetch_and_save
from src.train import train
from src.demo import run_demo


def main():
    print("=" * 60)
    print("  Auto Email / Ticket Categorizer")
    print("  Fobes Skill Itech — AI/ML Intern Assessment")
    print("=" * 60)

    # Step 1: Fetch dataset
    print("\n[1/3] Fetching dataset from HuggingFace...")
    csv_path = os.path.join("data", "tickets.csv")
    if not os.path.exists(csv_path):
        fetch_and_save(csv_path)
    else:
        print(f"  Dataset already exists at {csv_path}")

    # Step 2: Train model
    print("\n[2/3] Training classifier...")
    train(csv_path)

    # Step 3: Run demo
    print("\n[3/3] Running demo...")
    run_demo()


if __name__ == "__main__":
    main()
