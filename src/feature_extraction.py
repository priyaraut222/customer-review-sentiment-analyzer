"""Key feature (aspect) extraction using spaCy noun chunks, linked to sentence-level sentiment."""
import re
import warnings
from collections import Counter, defaultdict
 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
 
from src.preprocess import clean_text
 
_NLP = None
_BACKEND = None          # "spacy" or "fallback" once initialised
 
# Nouns that are too generic to be useful product features
GENERIC = {
    "thing", "things", "one", "ones", "time", "day", "days", "way", "lot", "bit", "kind", "sort",
    "review", "reviews", "product", "item", "everything", "anything", "something", "nothing",
    "money", "year", "years", "week", "weeks", "month", "months", "hour", "hours", "minute", "minutes",
    "person", "people", "guy", "order", "purchase", "buy", "experience", "hype", "fan",
}
 
# Opinion words the rule-based fallback must not mistake for features
_OPINION = {
    "great", "good", "bad", "terrible", "awful", "excellent", "amazing", "poor", "worst", "best", "love", "loved",
    "hate", "hated", "nice", "fine", "okay", "ok", "decent", "average", "horrible", "fantastic", "perfect",
    "disappointed", "disappointing", "happy", "unhappy", "slow", "fast", "quick", "late", "broken", "useless",
    "lovely", "brilliant", "wonderful", "mediocre", "overpriced", "cheap", "expensive", "really", "very", "quite",
    "just", "still", "never", "always", "also", "even", "much", "many", "got", "get", "gets", "getting",
    "bought", "buy", "buying", "ordered", "order", "tried", "try", "use", "used", "using", "worked", "work",
    "works", "working", "stopped", "arrived", "came", "come", "went", "take", "took", "make", "made",
    "multiple", "registered", "waited", "waiting", "turned", "froze", "sent", "told", "said", "asked", "called",
} | set(GENERIC)
_SKIP_WORDS = set(ENGLISH_STOP_WORDS) | _OPINION
_DETERMINER_PHRASE = re.compile(
    r"\b(?:the|this|that|these|those|my|our|your|his|her|its|their)\s+([a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?)", re.I)
_SENTENCE_START = re.compile(r"^\W*([a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?)", re.I)
 
 
def get_nlp():
    """Return the loaded spaCy pipeline, or None when the rule-based fallback is in use."""
    global _NLP, _BACKEND
    if _BACKEND is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
            nlp("The screen is great.")                  # smoke test: loading can succeed yet running still fail
            _NLP, _BACKEND = nlp, "spacy"
        except OSError as e:
            raise RuntimeError("spaCy model missing. Run: python -m spacy download en_core_web_sm") from e
        except Exception as e:                           # e.g. OverflowError from spaCy on an unsupported Python
            warnings.warn(f"spaCy could not run ({type(e).__name__}); using the simpler rule-based feature "
                          f"extractor instead. Use Python 3.10-3.12 for the full spaCy extractor.")
            _NLP, _BACKEND = None, "fallback"
    return _NLP
 
 
def backend_name() -> str:
    """'spacy' or 'fallback' (initialises the backend on first call)."""
    get_nlp()
    return _BACKEND
 
 
# --------------------------------------------------------------------------- spaCy path
def _chunk_to_feature(chunk):
    """Turn a spaCy noun chunk into a normalised feature string (e.g. 'battery life')."""
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
 
 
# --------------------------------------------------------------------------- rule-based fallback
def _split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]
 
 
def _trim(phrase: str):
    """Strip stop/opinion words from both ends of a candidate phrase; None if nothing useful is left."""
    words = phrase.lower().split()
    while words and words[0] in _SKIP_WORDS:
        words.pop(0)
    while words and words[-1] in _SKIP_WORDS:
        words.pop()
    if not words or any(w in _SKIP_WORDS for w in words):       # 'battery is' trimmed; 'battery and screen' rejected
        return None
    feature = " ".join(words)
    return feature if len(feature) >= 3 and feature not in GENERIC else None
 
 
def _fallback_features(sentence: str) -> list:
    """Candidate features: words after 'the/my/this...' and the opening words of the sentence."""
    found = []
    for m in _DETERMINER_PHRASE.finditer(sentence):
        found.append(_trim(m.group(1)))
    m = _SENTENCE_START.match(sentence)
    # only trust the opening words when the sentence does not start with a pronoun/determiner/verb
    # ("Delivery was slow" yes; "I registered on..." and "The battery..." no, the latter is handled above)
    if m and m.group(1).split()[0].lower() not in _SKIP_WORDS:
        found.append(_trim(m.group(1)))
    out = []
    for f in found:
        if f and f not in out:
            out.append(f)
    return out
 
 
# --------------------------------------------------------------------------- public API
def _sentences_with_features(text: str) -> list:
    """[(sentence_text, [features in that sentence])] using whichever backend is active."""
    nlp = get_nlp()
    if nlp is not None:
        doc = nlp(str(text))
        return [(s.text.strip(), [f for f in (_chunk_to_feature(c) for c in s.noun_chunks) if f])
                for s in doc.sents if s.text.strip()]
    return [(s, _fallback_features(s)) for s in _split_sentences(text)]
 
 
def extract_features(text: str) -> list:
    """Return the unique key features mentioned in a review."""
    seen, out = set(), []
    for _sent, feats in _sentences_with_features(text):
        for f in feats:
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out
 
 
def aspect_sentiments(model, text: str) -> list:
    """For each sentence, predict sentiment and attach it to the features in that sentence."""
    results = []
    for sent_text, feats in _sentences_with_features(text):
        if not feats:
            continue
        label = model.predict([clean_text(sent_text)])[0]
        for f in feats:
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