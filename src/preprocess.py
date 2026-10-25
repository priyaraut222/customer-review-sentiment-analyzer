
"""Text cleaning, rating parsing and CSV loading utilities."""
import re
 
import pandas as pd
 
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
 
 
def parse_rating(value):
    """Turn a rating into a float 1-5, or None if it can't be understood.
 
    Accepts numbers (4, 4.0, "4") and text such as "Rated 1 out of 5 stars" (the first number is used).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
    else:
        m = _NUMBER.search(str(value))
        if not m:
            return None
        num = float(m.group())
    return num if 1 <= num <= 5 else None
 
 
def rating_to_label(rating: float) -> str:
    """Map a 1-5 star rating to a sentiment label."""
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"
 
 
def clean_text(text: str) -> str:
    """Lower-case, strip HTML/URLs and non-letters (keeps apostrophes and '!')."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s'!]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
 
 
def read_csv_tolerant(path: str) -> pd.DataFrame:
    """Read a CSV, falling back to the slower forgiving reader if the file is messy.
 
    Scraped review files often contain stray quotes or rows with the wrong number of fields, which makes pandas'
    fast reader fail with "Buffer overflow" / "Error tokenizing data". The fallback skips only the broken rows.
    """
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", engine="python", on_bad_lines="skip")
    except pd.errors.ParserError:
        print("Note: the CSV has some malformed rows; using the tolerant reader (broken rows are skipped).")
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
 
 
def load_reviews(path: str, text_col: str = "review", rating_col: str = "rating") -> pd.DataFrame:
    """Load a CSV of reviews and add numeric `stars`, `label` and `clean` columns."""
    df = read_csv_tolerant(path)
    if text_col not in df.columns or rating_col not in df.columns:
        raise ValueError(f"CSV must contain '{text_col}' and '{rating_col}' columns. Found: {list(df.columns)}")
    df = df.dropna(subset=[text_col, rating_col]).copy()
    df["stars"] = df[rating_col].apply(parse_rating)
    dropped = int(df["stars"].isna().sum())
    if dropped:
        print(f"Note: dropped {dropped} rows whose rating could not be read as a number from 1 to 5.")
    df = df.dropna(subset=["stars"]).copy()
    if df.empty:
        raise ValueError(f"No usable rows: column '{rating_col}' has no ratings from 1 to 5.")
    df["label"] = df["stars"].apply(rating_to_label)
    df["clean"] = df[text_col].apply(clean_text)
    return df
 