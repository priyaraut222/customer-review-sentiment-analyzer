"""Sentiment classifier: TF-IDF + Logistic Regression baseline."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.preprocess import clean_text

LABELS = ["negative", "neutral", "positive"]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "report": classification_report(y_test, preds, labels=LABELS, digits=3),
        "report_dict": classification_report(y_test, preds, labels=LABELS, output_dict=True),
        "confusion": confusion_matrix(y_test, preds, labels=LABELS),
        "labels": LABELS,
    }


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def predict_sentiment(model: Pipeline, text: str) -> tuple[str, dict]:
    """Return (label, {label: probability}) for a single review."""
    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    label = classes[int(probs.argmax())]
    return label, {c: float(p) for c, p in zip(classes, probs)}
