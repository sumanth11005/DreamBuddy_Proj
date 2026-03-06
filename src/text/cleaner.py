import re
import unicodedata
from typing import List


def clean_caption(text: str, keep_hashtags: bool = True) -> str:
    """Full caption cleaning: remove URLs, mentions, normalise Unicode."""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    if not keep_hashtags:
        text = re.sub(r"#\w+", " ", text)
    text = unicodedata.normalize("NFKC", text)   # handles Roman-Hindi normalisation
    return re.sub(r"\s+", " ", text).strip()


def extract_hashtag_string(tags: List[str]) -> str:
    """Join hashtags into a single space-separated string for TF-IDF."""
    return " ".join(tags)


def build_combined_text(caption: str, tags: List[str]) -> str:
    """Concatenate cleaned caption + hashtags for vectorisation."""
    return f"{clean_caption(caption, keep_hashtags=False)} {extract_hashtag_string(tags)}".strip()
