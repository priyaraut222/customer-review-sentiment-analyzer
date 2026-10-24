"""Offline tests for the prompt-engineering layer (no model download needed: a MockLLM stands in)."""
import json

import pytest

from src import prompts as P
from src.llm import MockLLM
from src.prompt_analyzer import PromptAnalyzer

REVIEW = "The screen is lovely but the battery barely lasts four hours."


# ----------------------------------------------------------------------------- prompt construction
def test_zero_shot_is_naive():
    msgs = P.zero_shot(REVIEW)
    assert [m["role"] for m in msgs] == ["user"]
    assert "<review>" not in msgs[0]["content"] and P.GUARD not in msgs[0]["content"]


@pytest.mark.parametrize("technique", ["role", "few_shot", "cot", "structured"])
def test_guard_and_delimiters(technique):
    with_guard = P.TECHNIQUES[technique](REVIEW, guard=True)
    no_guard = P.TECHNIQUES[technique](REVIEW, guard=False)
    assert P.GUARD in with_guard[0]["content"]
    assert P.GUARD not in no_guard[0]["content"]
    assert "<review>" in with_guard[-1]["content"]          # untrusted text is delimited


def test_few_shot_structure():
    msgs = P.few_shot(REVIEW)
    assert msgs[0]["role"] == "system"
    roles = [m["role"] for m in msgs[1:]]
    assert roles == ["user", "assistant"] * len(P.FEW_SHOT_EXAMPLES) + ["user"]
    assert REVIEW in msgs[-1]["content"]
    assert {m["content"] for m in msgs if m["role"] == "assistant"} <= set(P.LABELS)


def test_few_shot_k_limits_examples():
    assert len(P.few_shot(REVIEW, k=2)) == 1 + 2 * 2 + 1


def test_structured_has_schema_and_oneshot_example():
    msgs = P.structured(REVIEW)
    assert "overall_sentiment" in msgs[1]["content"]
    assert json.loads(msgs[2]["content"])["overall_sentiment"] in P.LABELS


# ----------------------------------------------------------------------------- parsers
@pytest.mark.parametrize("text,prefer,expected", [
    ("Positive.", "first", "positive"),
    ("negative", "first", "negative"),
    ("It is Neutral overall", "first", "neutral"),
    ("Final answer: **Negative**", "last", "negative"),
    ("Screen is positive, battery is negative.\nFinal answer: neutral", "last", "neutral"),
    ("positives and negatives mixed, so maybe positive", "last", "positive"),
    ("I cannot tell", "first", None),
])
def test_parse_label(text, prefer, expected):
    assert P.parse_label(text, prefer) == expected


def test_parse_label_ignores_substrings():
    assert P.parse_label("unpositive vibes") is None


def test_extract_json_handles_fences_prose_and_braces_in_strings():
    text = 'Sure! ```json\n{"a": "x } y", "b": {"c": 1}}\n``` hope that helps'
    assert json.loads(P.extract_json_object(text)) == {"a": "x } y", "b": {"c": 1}}


def test_extract_json_errors():
    with pytest.raises(P.ParseError):
        P.extract_json_object("no braces here")
    with pytest.raises(P.ParseError):
        P.extract_json_object('{"a": 1')


def test_parse_structured_valid_and_normalises():
    raw = ('```json\n{"overall_sentiment": "Negative", "aspects": ['
           '{"feature": " Battery ", "sentiment": "NEGATIVE"}, '
           '{"feature": "screen", "sentiment": "great"}, "junk"], "summary": " ok "}\n```')
    out = P.parse_structured(raw)
    assert out["overall_sentiment"] == "negative"
    assert out["aspects"] == [{"feature": "battery", "sentiment": "negative"}]   # invalid aspects dropped
    assert out["summary"] == "ok"


@pytest.mark.parametrize("raw", ['{"overall_sentiment": "great"}', '{"aspects": []}', "{bad json}", "plain text"])
def test_parse_structured_rejects(raw):
    with pytest.raises(P.ParseError):
        P.parse_structured(raw)


def test_parse_features_and_summary():
    assert P.parse_features('{"features": ["Battery", "battery", "screen", 3]}') == ["battery", "screen"]
    assert P.parse_features("nonsense") == []
    s = P.parse_chain_summary("Overall: Negative\nSummary: Unhappy with battery.\nAction: Improve battery.")
    assert s == {"overall": "negative", "summary": "Unhappy with battery.", "action": "Improve battery."}


# ----------------------------------------------------------------------------- analyzer logic
def test_single_label_techniques_parse_and_flag_invalid():
    a = PromptAnalyzer(MockLLM(["Positive."]))
    r = a.few_shot(REVIEW)
    assert (r.overall, r.ok) == ("positive", True)
    bad = PromptAnalyzer(MockLLM(["I am not sure."])).role(REVIEW)
    assert (bad.overall, bad.ok) == (None, False)


def test_cot_uses_final_answer_line():
    raw = "screen: positive\nbattery: negative\nBalanced, so...\nFinal answer: neutral"
    r = PromptAnalyzer(MockLLM([raw])).cot(REVIEW)
    assert r.overall == "neutral"


def test_self_consistency_majority_and_agreement():
    samples = ["Final answer: negative"] * 3 + ["Final answer: neutral", "Final answer: positive"]
    llm = MockLLM([samples])
    r = PromptAnalyzer(llm).self_consistency(REVIEW, n=5, temperature=0.7)
    assert r.overall == "negative"
    assert r.extra["agreement"] == pytest.approx(0.6)
    assert r.extra["votes"] == {"negative": 3, "neutral": 1, "positive": 1}
    assert llm.calls[0]["temperature"] == 0.7 and llm.calls[0]["n"] == 5     # actually sampled


def test_self_consistency_all_unparsable():
    r = PromptAnalyzer(MockLLM([["???"] * 3])).self_consistency(REVIEW, n=3)
    assert r.overall is None and not r.ok


GOOD_JSON_BODY = ('"overall_sentiment": "neutral", "aspects": [{"feature": "screen", "sentiment": "positive"}, '
                  '{"feature": "battery", "sentiment": "negative"}], "summary": "Mixed."}')


def test_structured_primes_with_brace():
    llm = MockLLM([GOOD_JSON_BODY])
    r = PromptAnalyzer(llm).structured(REVIEW)
    assert llm.calls[0]["prefill"] == "{"
    assert r.overall == "neutral" and len(r.aspects) == 2 and r.ok


def test_structured_repair_loop_recovers():
    llm = MockLLM(['"overall_sentiment": "meh"}', GOOD_JSON_BODY])
    r = PromptAnalyzer(llm).structured(REVIEW, retries=1)
    assert r.ok and r.overall == "neutral" and r.extra["retries_used"] == 1
    repair_prompt = llm.calls[1]["messages"]
    assert repair_prompt[-2]["role"] == "assistant"                     # the bad output is shown back
    assert "overall_sentiment" in repair_prompt[-1]["content"] or "corrected" in repair_prompt[-1]["content"]
    assert "'overall_sentiment' must be one of" in repair_prompt[-1]["content"]   # with the parse error


def test_structured_gives_up_after_retries():
    r = PromptAnalyzer(MockLLM(["not json at all"])).structured(REVIEW, retries=1)
    assert not r.ok and r.overall is None and len(r.raw) == 2


def test_chain_feeds_each_step_into_the_next():
    llm = MockLLM([
        '"features": ["screen", "battery"]}',          # step 1 (prefilled with "{")
        "positive",                                    # step 2: screen
        "negative",                                    # step 2: battery
        "Overall: neutral\nSummary: Mixed.\nAction: Fix battery.",   # step 3
    ])
    r = PromptAnalyzer(llm).chain(REVIEW)
    assert r.overall == "neutral" and r.ok
    assert r.aspects == [{"feature": "screen", "sentiment": "positive"},
                         {"feature": "battery", "sentiment": "negative"}]
    assert r.extra["action"] == "Fix battery."
    step3_prompt = llm.calls[3]["messages"][-1]["content"]
    assert "screen: positive" in step3_prompt and "battery: negative" in step3_prompt   # chained context
    assert "'battery'" in llm.calls[2]["messages"][-1]["content"]                      # per-feature prompt


def test_chain_falls_back_to_rule_when_summary_unparsable():
    llm = MockLLM(['"features": ["screen", "battery"]}', "positive", "negative", "gibberish"])
    assert PromptAnalyzer(llm).chain(REVIEW).overall == "neutral"        # balanced -> neutral


def test_chain_with_no_features():
    llm = MockLLM(['"features": []}', "Overall: neutral\nSummary: s\nAction: a"])
    r = PromptAnalyzer(llm).chain("ok")
    assert r.aspects == [] and r.overall == "neutral"


def test_guard_flag_reaches_the_prompt():
    llm = MockLLM(["positive"])
    PromptAnalyzer(llm, guard=False).few_shot(REVIEW)
    assert P.GUARD not in llm.calls[0]["messages"][0]["content"]
    llm2 = MockLLM(["positive"])
    PromptAnalyzer(llm2, guard=True).few_shot(REVIEW)
    assert P.GUARD in llm2.calls[0]["messages"][0]["content"]


def test_run_dispatcher_rejects_unknown():
    with pytest.raises(ValueError):
        PromptAnalyzer(MockLLM(["x"])).run("telepathy", REVIEW)
