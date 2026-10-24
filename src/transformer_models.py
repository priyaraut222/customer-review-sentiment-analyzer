"""DistilBERT sentiment classifier (Hugging Face Transformers): fine-tuning + inference.

Plain PyTorch loop on purpose: short, readable, and independent of Trainer API changes between versions.
"""
import os
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]          # class index order used by the model head
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
DEFAULT_BASE = "distilbert-base-uncased"


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class DistilBertClassifier:
    """Load a fine-tuned checkpoint directory and predict labels."""

    def __init__(self, model_dir: str, device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device or _device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def predict_proba(self, texts, batch_size: int = 32, max_len: int = 128):
        import torch
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(list(texts[i:i + batch_size]), truncation=True, max_length=max_len,
                                 padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.id2label)))

    def predict(self, texts, **kw):
        probs = self.predict_proba(texts, **kw)
        return [self.id2label[int(i)] for i in probs.argmax(axis=1)]


def finetune(texts, labels, base: str = DEFAULT_BASE, out_dir: str = "models/distilbert", epochs: int = 2,
             batch_size: int = 16, lr: float = 5e-5, max_len: int = 128, test_size: float = 0.2,
             seed: int = 42, max_samples: int | None = None, log=print) -> dict:
    """Fine-tune `base` on (texts, labels) and save the best-effort final model to `out_dir`.

    Uses the *raw* review text (BERT's tokenizer expects natural text, not the cleaned/stemmed form that the
    TF-IDF baseline uses). Class-weighted cross-entropy mirrors the baseline's class_weight='balanced'.
    Returns the held-out evaluation report.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    texts, labels = list(texts), list(labels)
    if max_samples and len(texts) > max_samples:
        idx = np.random.RandomState(seed).choice(len(texts), max_samples, replace=False)
        texts, labels = [texts[i] for i in idx], [labels[i] for i in idx]

    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=test_size, stratify=labels, random_state=seed)
    device = _device()
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID).to(device)

    def encode(batch_texts):
        return tok(batch_texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")

    y_tr_ids = np.array([LABEL2ID[l] for l in y_tr])
    counts = np.bincount(y_tr_ids, minlength=len(LABELS)).astype(float)
    weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    steps_per_epoch = (len(X_tr) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)

    log(f"Fine-tuning {base} on {len(X_tr)} reviews ({len(X_te)} held out) | device={device} | {total_steps} steps")
    start = time.time()
    for epoch in range(epochs):
        model.train()
        order = np.random.RandomState(seed + epoch).permutation(len(X_tr))
        running = 0.0
        for step in range(steps_per_epoch):
            ids = order[step * batch_size:(step + 1) * batch_size]
            enc = encode([X_tr[i] for i in ids]).to(device)
            target = torch.tensor(y_tr_ids[ids], device=device)
            loss = loss_fn(model(**enc).logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            running += loss.item()
            if (step + 1) % 50 == 0 or step + 1 == steps_per_epoch:
                log(f"  epoch {epoch + 1}/{epochs} step {step + 1}/{steps_per_epoch} loss {running / (step + 1):.4f}")
    log(f"Training took {time.time() - start:.0f}s")

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    clf = DistilBertClassifier(out_dir, device=device)
    preds = clf.predict(X_te, max_len=max_len)
    return {
        "report": classification_report(y_te, preds, labels=LABELS, digits=3, zero_division=0),
        "report_dict": classification_report(y_te, preds, labels=LABELS, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_te, preds, labels=LABELS),
        "labels": LABELS,
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }
