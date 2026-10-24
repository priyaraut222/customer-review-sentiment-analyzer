"""Fine-tune DistilBERT for 3-class review sentiment.

Usage:
    python train_distilbert.py --data reviews.csv --text-col Text --rating-col Score --max-samples 6000
    python train_distilbert.py --help

Needs internet access to download `distilbert-base-uncased` from the Hugging Face Hub (first run only).
On a CPU use --max-samples 3000-6000 (about 10-30 min); on a free Colab GPU you can use the full dataset.
"""
import argparse
import json
import os

from src.preprocess import load_reviews
from src.transformer_models import DEFAULT_BASE, finetune


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample_reviews.csv")
    p.add_argument("--text-col", default="review")
    p.add_argument("--rating-col", default="rating")
    p.add_argument("--base", default=DEFAULT_BASE, help="Hub id or local directory of the base model")
    p.add_argument("--out", default="models/distilbert")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    df = load_reviews(args.data, args.text_col, args.rating_col)
    print(f"Loaded {len(df)} reviews. Class balance:\n{df['label'].value_counts()}\n")

    res = finetune(df[args.text_col].astype(str).tolist(), df["label"].tolist(), base=args.base, out_dir=args.out,
                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_len=args.max_len,
                   max_samples=args.max_samples)
    print(res["report"])
    print("Confusion matrix (rows=true, cols=pred):", res["labels"])
    print(res["confusion"])

    os.makedirs("models", exist_ok=True)
    with open("models/distilbert_metrics.json", "w") as f:
        json.dump(res["report_dict"], f, indent=2)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
