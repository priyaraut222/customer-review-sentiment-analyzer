"""Runs the real DistilBERT fine-tuning code path with NO network:
a tiny randomly-initialised DistilBERT + a tokenizer trained locally stand in for `distilbert-base-uncased`.
This checks the training loop, saving/loading and prediction plumbing, not model quality."""
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.transformer_models import LABELS, DistilBertClassifier, finetune


def build_tiny_base(path, texts):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast

    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(texts, trainers.WordPieceTrainer(
        vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]", model_input_names=["input_ids", "attention_mask"])
    cfg = DistilBertConfig(vocab_size=len(fast), dim=32, hidden_dim=64, n_layers=1, n_heads=2,
                           max_position_embeddings=64)
    DistilBertModel(cfg).save_pretrained(path)       # encoder only, like the real base checkpoint (no head)
    fast.save_pretrained(path)


def make_data(n_per_class=24):
    pos = ["great battery and a lovely screen", "love it works perfectly", "excellent quality very happy"]
    neg = ["terrible battery broke quickly", "hate it stopped working", "awful quality very disappointed"]
    neu = ["it is okay nothing special", "average product does the job", "fine but not great"]
    texts, labels = [], []
    for i in range(n_per_class):
        texts += [pos[i % 3] + f" {i}", neg[i % 3] + f" {i}", neu[i % 3] + f" {i}"]
        labels += ["positive", "negative", "neutral"]
    return texts, labels


def test_finetune_save_load_predict(tmp_path):
    texts, labels = make_data()
    base, out = str(tmp_path / "base"), str(tmp_path / "out")
    build_tiny_base(base, texts)

    logs = []
    res = finetune(texts, labels, base=base, out_dir=out, epochs=2, batch_size=8, lr=1e-3, max_len=16,
                   log=logs.append)

    assert res["labels"] == LABELS and res["n_train"] + res["n_test"] == len(texts)
    assert res["confusion"].shape == (3, 3) and "accuracy" in res["report_dict"]
    assert any("loss" in line for line in logs)
    assert os.path.exists(os.path.join(out, "config.json"))

    clf = DistilBertClassifier(out)
    preds = clf.predict(["great screen love it", "terrible and awful"], max_len=16)
    assert len(preds) == 2 and set(preds) <= set(LABELS)
    probs = clf.predict_proba(["okay i guess"], max_len=16)
    assert probs.shape == (1, 3) and probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_finetune_max_samples_subsamples(tmp_path):
    texts, labels = make_data()
    base = str(tmp_path / "base")
    build_tiny_base(base, texts)
    res = finetune(texts, labels, base=base, out_dir=str(tmp_path / "o"), epochs=1, batch_size=8, max_len=16,
                   max_samples=30, log=lambda *_: None)
    assert res["n_train"] + res["n_test"] == 30
