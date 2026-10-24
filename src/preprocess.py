
"""Text cleaning and label utilities."""

import re

import pandas as pd


def rating_to_label(rating) -> str | None:
    """Map a 1-5 star rating to a sentiment label."""
    match = re.search(r"(\d+(?:\.\d+)?)", str(rating))

    if not match:
        return None

    rating = float(match.group(1))

    if rating < 1 or rating > 5:
        return None
    elif rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def clean_text(text: str) -> str:
    """Lowercase text and remove HTML, URLs, and unwanted characters."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s'!]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_reviews(
    path: str,
    text_col: str = "review",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """Load reviews, generate sentiment labels, and clean review text."""

    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="warn",
    )

    if text_col not in df.columns or rating_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{text_col}' and '{rating_col}' columns. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_col, rating_col]).copy()

    # Convert textual ratings into sentiment labels.
    df["label"] = df[rating_col].apply(rating_to_label)

    # Remove rows with invalid ratings.
    df = df.dropna(subset=["label"]).copy()

    # Clean review text.
    df["clean"] = df[text_col].apply(clean_text)

    # Remove empty reviews.
    df = df[df["clean"].str.strip().ne("")].copy()

    return df

