"""Train and evaluate the sentiment classifier.

Usage:
    python train.py                                   # uses data/sample_reviews.csv
    python train.py --data my_reviews.csv --text-col Text --rating-col Score
"""
import argparse
import json
import os

from sklearn.model_selection import train_test_split

from src.preprocess import load_reviews
from src.sentiment_model import build_pipeline, evaluate, save_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample_reviews.csv")
    p.add_argument("--text-col", default="review")
    p.add_argument("--rating-col", default="rating")
    p.add_argument("--out", default="models/sentiment_model.joblib")
    args = p.parse_args()

    df = load_reviews(args.data, args.text_col, args.rating_col)
    print(f"Loaded {len(df)} reviews. Class balance:\n{df['label'].value_counts()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    model = build_pipeline().fit(X_train, y_train)
    res = evaluate(model, X_test, y_test)
    print(res["report"])
    print("Confusion matrix (rows=true, cols=pred):", res["labels"])
    print(res["confusion"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_model(model, args.out)
    with open("models/metrics.json", "w") as f:
        json.dump(res["report_dict"], f, indent=2)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
