"""Compare the TF-IDF baseline, fine-tuned DistilBERT and every prompting technique on one labelled eval set.

Usage:
    python compare.py                                    # everything that is available
    python compare.py --methods baseline,zero_shot,few_shot,cot --limit 12     # quick run
    python compare.py --llm Qwen/Qwen2.5-0.5B-Instruct   # smaller / faster model
    python compare.py --eval my_labelled_reviews.csv     # columns: review,label,aspects(json),adversarial

Outputs (folder results/):
    comparison.csv / comparison.md   one row per method
    predictions.csv                  per-review predictions (for error analysis)
    raw_outputs.jsonl                the raw LLM text for every review (for prompt iteration)

Metrics
    accuracy, macro_f1      overall sentiment vs gold label
    adversarial_acc         accuracy on the prompt-injection reviews only (robustness)
    valid_rate              share of replies that followed the requested output format
    aspect_f1               feature extraction vs gold (fuzzy string match), only for methods that output features
    aspect_sent_acc         per-feature sentiment accuracy on correctly found features
    sec_per_review          mean latency (depends heavily on your hardware)
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.prompts import LABELS

LLM_METHODS = ["zero_shot", "role", "few_shot", "few_shot_noguard", "cot", "self_consistency", "structured", "chain"]
ALL_METHODS = ["baseline", "distilbert"] + LLM_METHODS


# ----------------------------------------------------------------------------- aspect scoring
def _norm(f: str) -> str:
    f = re.sub(r"[^a-z ]", "", str(f).lower()).strip()
    return f[:-1] if len(f) > 3 and f.endswith("s") and not f.endswith("ss") else f


def feature_match(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def aspect_counts(pred_aspects: list, gold: dict):
    """Greedy one-to-one matching. Returns (true_positives, n_pred, n_gold, sentiment_correct)."""
    pool, tp, correct = list(pred_aspects), 0, 0
    for g_feat, g_sent in gold.items():
        for j, p in enumerate(pool):
            if feature_match(p["feature"], g_feat):
                tp += 1
                correct += int(p["sentiment"] == g_sent)
                pool.pop(j)
                break
    return tp, len(pred_aspects), len(gold), correct


# ----------------------------------------------------------------------------- method runners
# every runner: review -> (label or None, aspects list or None, followed_format, raw_text)
def make_baseline(args):
    from src.feature_extraction import aspect_sentiments
    from src.preprocess import clean_text
    from src.sentiment_model import load_model
    model = load_model(args.baseline)

    def run(review):
        label = model.predict([clean_text(review)])[0]
        seen = {}
        for it in aspect_sentiments(model, review):
            seen.setdefault(it["feature"], it["sentiment"])
        return label, [{"feature": f, "sentiment": s} for f, s in seen.items()], True, ""
    return run


def make_distilbert(args):
    from src.transformer_models import DistilBertClassifier
    clf = DistilBertClassifier(args.distilbert)
    return lambda review: (clf.predict([review])[0], None, True, "")


def make_llm_runner(method, llm, args):
    from src.prompt_analyzer import PromptAnalyzer
    technique = "few_shot" if method == "few_shot_noguard" else method
    analyzer = PromptAnalyzer(llm, guard=(method != "few_shot_noguard"))
    kw = {"n": args.sc_n} if technique == "self_consistency" else {}

    def run(review):
        r = analyzer.run(technique, review, **kw)
        aspects = r.aspects if technique in ("structured", "chain") else None
        return r.overall, aspects, r.ok, " || ".join(r.raw)
    return run


# ----------------------------------------------------------------------------- evaluation
def evaluate(method, runner, df, gold_aspects):
    preds, oks, raws, times, asp_preds = [], [], [], [], []
    for i, review in enumerate(df["review"], 1):
        t0 = time.time()
        label, aspects, ok, raw = runner(review)
        times.append(time.time() - t0)
        preds.append(label or "invalid")
        oks.append(ok)
        raws.append(raw)
        asp_preds.append(aspects)
        if i % 10 == 0:
            print(f"    {method}: {i}/{len(df)}")

    y = df["label"].tolist()
    adv = (df["adversarial"] == 1).to_numpy()
    row = {
        "method": method,
        "accuracy": accuracy_score(y, preds),
        # average only over classes that exist in the gold labels (a subset may lack one)
        "macro_f1": f1_score(y, preds, labels=[l for l in LABELS if l in set(y)], average="macro", zero_division=0),
        "adversarial_acc": (accuracy_score(np.array(y)[adv], np.array(preds)[adv]) if adv.any() else np.nan),
        "valid_rate": float(np.mean(oks)),
        "aspect_f1": np.nan,
        "aspect_sent_acc": np.nan,
        "sec_per_review": float(np.mean(times)),
        "n": len(df),
    }
    if all(a is not None for a in asp_preds):
        tp = n_pred = n_gold = correct = 0
        for aspects, gold in zip(asp_preds, gold_aspects):
            a, b, c, d = aspect_counts(aspects, gold)
            tp, n_pred, n_gold, correct = tp + a, n_pred + b, n_gold + c, correct + d
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gold if n_gold else 0.0
        row["aspect_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        row["aspect_sent_acc"] = correct / tp if tp else 0.0
    return row, preds, raws


def balanced_subset(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """First `n` rows taken round-robin across labels (the eval CSV is ordered by class, so head(n) would
    contain a single class and make quick runs meaningless)."""
    order = df.assign(_rank=df.groupby("label").cumcount()).sort_values(["_rank", "id"])
    return order.head(n).drop(columns="_rank").sort_values("id").reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fmt = lambda v: "-" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", default="data/eval_reviews.csv")
    p.add_argument("--methods", default=",".join(ALL_METHODS), help=f"comma list from: {', '.join(ALL_METHODS)}")
    p.add_argument("--llm", default=None, help="Hub id of an instruction-tuned chat model (default: $HF_LLM or Qwen2.5-1.5B-Instruct)")
    p.add_argument("--baseline", default="models/sentiment_model.joblib")
    p.add_argument("--distilbert", default="models/distilbert")
    p.add_argument("--sc-n", type=int, default=5, help="samples for self-consistency")
    p.add_argument("--limit", type=int, default=None, help="only the first N eval reviews (quick test)")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    df = pd.read_csv(args.eval)
    if args.limit:
        df = balanced_subset(df, args.limit)
    gold_aspects = [json.loads(a) for a in df["aspects"]]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Choose from {ALL_METHODS}")
    os.makedirs(args.out, exist_ok=True)
    print(f"Evaluating {len(methods)} methods on {len(df)} reviews "
          f"({int((df['adversarial'] == 1).sum())} adversarial)\n")

    llm = None
    rows, pred_table, raw_log = [], {"id": df["id"].tolist(), "true": df["label"].tolist()}, []
    for method in methods:
        print(f"== {method}")
        try:
            if method == "baseline":
                runner = make_baseline(args)
            elif method == "distilbert":
                runner = make_distilbert(args)
            else:
                if llm is None:
                    from src.llm import DEFAULT_MODEL, HFChatLLM
                    name = args.llm or DEFAULT_MODEL
                    print(f"  loading LLM {name} ...")
                    llm = HFChatLLM(name)
                runner = make_llm_runner(method, llm, args)
            row, preds, raws = evaluate(method, runner, df, gold_aspects)
        except Exception as e:  # keep going: one unavailable method should not kill the whole comparison
            print(f"  SKIPPED ({type(e).__name__}): {str(e)[:300]}")
            continue
        rows.append(row)
        pred_table[method] = preds
        raw_log += [{"method": method, "id": int(i), "raw": r} for i, r in zip(df["id"], raws) if r]
        print(f"  accuracy {row['accuracy']:.3f} | macro-F1 {row['macro_f1']:.3f} | valid {row['valid_rate']:.2f}")

    if not rows:
        raise SystemExit("No method could be evaluated.")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "comparison.csv"), index=False)
    with open(os.path.join(args.out, "comparison.md"), "w") as f:
        f.write(to_markdown(res) + "\n")
    pd.DataFrame(pred_table).to_csv(os.path.join(args.out, "predictions.csv"), index=False)
    with open(os.path.join(args.out, "raw_outputs.jsonl"), "w") as f:
        for item in raw_log:
            f.write(json.dumps(item) + "\n")
    print("\n" + to_markdown(res))
    print(f"\nSaved results to {args.out}/")


if __name__ == "__main__":
    main()
