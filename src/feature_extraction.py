"""Key feature (aspect) extraction using spaCy noun chunks, linked to sentence-level sentiment."""
from collections import Counter, defaultdict

import spacy

from src.preprocess import clean_text

_NLP = None

# Nouns that are too generic to be useful product features
GENERIC = {
    "thing", "things", "one", "ones", "time", "day", "days", "way", "lot", "bit", "kind", "sort",
    "review", "reviews", "product", "item", "everything", "anything", "something", "nothing",
    "money", "year", "years", "week", "weeks", "month", "months", "hour", "hours", "minute", "minutes",
    "person", "people", "guy", "order", "purchase", "buy", "experience", "hype", "fan",
}


def get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError as e:
            raise RuntimeError(
                "spaCy model missing. Run: python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


def _chunk_to_feature(chunk):
    """Turn a noun chunk into a normalised feature string (e.g. 'battery life')."""
    tokens = [t for t in chunk if t.pos_ in ("NOUN", "PROPN", "ADJ") and not t.is_stop]
    # drop leading adjectives like 'great'/'terrible' so we keep the noun phrase
    while tokens and tokens[0].pos_ == "ADJ":
        tokens = tokens[1:]
    if not tokens or tokens[-1].pos_ not in ("NOUN", "PROPN"):
        return None
    feature = " ".join(t.lemma_.lower() for t in tokens)
    if feature in GENERIC or len(feature) < 3:
        return None
    return feature


def extract_features(text: str) -> list:
    """Return the unique key features mentioned in a review."""
    doc = get_nlp()(str(text))
    seen, out = set(), []
    for chunk in doc.noun_chunks:
        f = _chunk_to_feature(chunk)
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def aspect_sentiments(model, text: str) -> list:
    """For each sentence, predict sentiment and attach it to the features in that sentence."""
    doc = get_nlp()(str(text))
    results = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        label = model.predict([clean_text(sent_text)])[0]
        for chunk in sent.noun_chunks:
            f = _chunk_to_feature(chunk)
            if f:
                results.append({"feature": f, "sentiment": label, "sentence": sent_text})
    return results


def summarize_features(model, reviews: list, top_n: int = 10) -> dict:
    """Aggregate aspect sentiment across many reviews.

    Returns {feature: {"positive": n, "negative": n, "neutral": n, "total": n}} for the top_n most-mentioned features.
    """
    stats = defaultdict(Counter)
    for review in reviews:
        seen = set()
        for item in aspect_sentiments(model, review):
            key = (item["feature"], item["sentiment"])
            if key in seen:
                continue
            seen.add(key)
            stats[item["feature"]][item["sentiment"]] += 1
    ranked = sorted(stats.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:top_n]
    return {
        f: {"positive": c["positive"], "negative": c["negative"], "neutral": c["neutral"], "total": sum(c.values())}
        for f, c in ranked
    }
