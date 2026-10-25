"""Tests for the evaluation harness (aspect scoring + end-to-end metrics with a scripted LLM)."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import compare
from src.eval_data import to_dataframe
from src.llm import MockLLM


def test_feature_match_is_fuzzy_but_not_silly():
    assert compare.feature_match("battery life", "battery")
    assert compare.feature_match("Screens", "screen")                # simple trailing-s plurals are handled
    assert not compare.feature_match("batteries", "battery")         # irregular plurals are NOT (known limitation)
    assert not compare.feature_match("screen", "keyboard")
    assert not compare.feature_match("", "screen")


def test_aspect_counts_one_to_one_matching():
    gold = {"battery": "negative", "screen": "positive", "price": "neutral"}
    pred = [{"feature": "battery life", "sentiment": "negative"},     # match, sentiment right
            {"feature": "screen", "sentiment": "negative"},           # match, sentiment wrong
            {"feature": "shipping", "sentiment": "positive"}]         # no gold counterpart
    assert compare.aspect_counts(pred, gold) == (2, 3, 3, 1)


def test_limit_gives_a_balanced_subset_not_one_class():
    df = to_dataframe()
    sub = compare.balanced_subset(df, 12)
    assert len(sub) == 12 and sub["label"].value_counts().to_dict() == {"positive": 4, "negative": 4, "neutral": 4}
    assert sub["id"].is_monotonic_increasing


def test_eval_set_is_well_formed():
    df = to_dataframe()
    assert len(df) == 40 and set(df["label"]) == {"positive", "neutral", "negative"}
    assert int(df["adversarial"].sum()) == 4
    assert all(len(json.loads(a)) >= 1 for a in df["aspects"])
    assert df["review"].is_unique


def test_markdown_table_renders_nan_as_dash():
    md = compare.to_markdown(pd.DataFrame([{"method": "x", "accuracy": 0.5, "aspect_f1": float("nan")}]))
    assert "| x | 0.500 | - |" in md


def test_end_to_end_metrics_with_scripted_llm():
    df = to_dataframe().head(8)
    gold = [json.loads(a) for a in df["aspects"]]
    truth = dict(zip(df["review"], df["label"]))

    def oracle(messages, temperature, n, prefill):         # a perfect "model": looks up the true label
        review = messages[-1]["content"]
        label = next((v for k, v in truth.items() if k in review), "neutral")
        return [label] * n

    runner = compare.make_llm_runner("few_shot", MockLLM(oracle), SimpleNamespace(sc_n=3))
    row, preds, _ = compare.evaluate("few_shot", runner, df, gold)
    assert row["accuracy"] == 1.0
    assert row["macro_f1"] == pytest.approx(1.0)
    assert row["valid_rate"] == 1.0 and preds == df["label"].tolist()
    assert pd.isna(row["aspect_f1"])                        # few_shot does not output features


def test_invalid_outputs_count_as_wrong_and_lower_valid_rate():
    df = to_dataframe().head(6)
    runner = compare.make_llm_runner("role", MockLLM(["I don't know"]), SimpleNamespace(sc_n=3))
    row, preds, _ = compare.evaluate("role", runner, df, [json.loads(a) for a in df["aspects"]])
    assert row["accuracy"] == 0.0 and row["valid_rate"] == 0.0 and set(preds) == {"invalid"}


def test_structured_method_reports_aspect_metrics():
    df = to_dataframe().head(1)                             # review 1: battery/keyboard/screen all positive
    body = ('"overall_sentiment": "positive", "aspects": [{"feature": "battery life", "sentiment": "positive"}, '
            '{"feature": "screen", "sentiment": "positive"}, {"feature": "price", "sentiment": "positive"}], '
            '"summary": "s"}')
    runner = compare.make_llm_runner("structured", MockLLM([body]), SimpleNamespace(sc_n=3))
    row, _, _ = compare.evaluate("structured", runner, df, [json.loads(a) for a in df["aspects"]])
    # gold = {battery, keyboard, screen}; pred = {battery life, screen, price} -> tp=2, P=2/3, R=2/3
    assert row["aspect_f1"] == pytest.approx(2 / 3)
    assert row["aspect_sent_acc"] == 1.0
