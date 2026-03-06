import math
import re
from datetime import datetime
from typing import Optional

from src.config import InstagramPostRecord

_FOOD_EMOJIS    = {"🍛", "🧈", "🌶️", "🥘", "🍲", "🫕", "😋", "🤤"}
_PAVBHAJI_TAGS  = {"pavbhaji", "pavbhajilover", "ekplatepavbhaji", "pavbhajirecipe"}
_CONFUSION_TAGS = {"vadapav", "masalapav", "dalkhichdi", "poha", "upma", "misal"}
_MUMBAI_TAGS    = {"mumbai", "mumbaikar", "mumbaifoodie", "mumbaistreetfood"}
_FOOD_TAGS      = {"food", "foodie", "foodporn", "instafood", "indianfood", "streetfood"}

_ZERO: dict = {
    "caption_raw": "", "caption_length": 0, "word_count": 0,
    "mentions_pavbhaji": 0, "mentions_butter": 0, "mentions_recipe": 0,
    "has_hindi_text": 0, "emoji_food_count": 0, "exclamation_count": 0,
    "has_pavbhaji_tag": 0, "has_confusion_tag": 0, "has_mumbai_tag": 0,
    "has_food_tag": 0, "total_tags": 0, "pavbhaji_tag_ratio": 0.0,
    "log_likes": 0.0, "log_comments": 0.0,
    "hour_of_day": -1, "day_of_week": -1, "month": -1, "is_weekend": 0,
    "has_location": 0, "aspect_ratio": 1.0, "is_square": 1, "json_is_video": 0,
}


def _clean_caption(text: str) -> str:
    """Strip URLs and collapse whitespace from a raw caption."""
    text = re.sub(r"http\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_all_features(rec: Optional[InstagramPostRecord]) -> dict:
    """Extract the full feature dict from one InstagramPostRecord (or None)."""
    if rec is None:
        return _ZERO.copy()

    caption  = rec.caption_text
    cap_low  = caption.lower()
    tag_set  = set(rec.tags)
    dt       = datetime.utcfromtimestamp(rec.taken_at_timestamp)
    pav_tags = tag_set & _PAVBHAJI_TAGS
    n_tags   = len(tag_set)

    return {
        # Text
        "caption_raw"       : _clean_caption(caption),
        "caption_length"    : len(caption),
        "word_count"        : len(caption.split()),
        "mentions_pavbhaji" : int("pav bhaji" in cap_low or "pavbhaji" in cap_low),
        "mentions_butter"   : int("butter" in cap_low or "makhan" in cap_low),
        "mentions_recipe"   : int("recipe" in cap_low or "ingredient" in cap_low),
        "has_hindi_text"    : int(bool(re.search(r"[\u0900-\u097F]", caption))),
        "emoji_food_count"  : sum(1 for ch in caption if ch in _FOOD_EMOJIS),
        "exclamation_count" : caption.count("!"),
        # Hashtags
        "has_pavbhaji_tag"  : int(bool(pav_tags)),
        "has_confusion_tag" : int(bool(tag_set & _CONFUSION_TAGS)),
        "has_mumbai_tag"    : int(bool(tag_set & _MUMBAI_TAGS)),
        "has_food_tag"      : int(bool(tag_set & _FOOD_TAGS)),
        "total_tags"        : n_tags,
        "pavbhaji_tag_ratio": round(len(pav_tags) / max(1, n_tags), 4),
        # Engagement
        "log_likes"         : math.log1p(rec.like_count),
        "log_comments"      : math.log1p(rec.comment_count),
        # Temporal
        "hour_of_day"       : dt.hour,
        "day_of_week"       : dt.weekday(),
        "month"             : dt.month,
        "is_weekend"        : int(dt.weekday() >= 5),
        # Context
        "has_location"      : int(rec.location is not None),
        "aspect_ratio"      : round(rec.dimensions.height / rec.dimensions.width, 4),
        "is_square"         : int(rec.dimensions.height == rec.dimensions.width),
        "json_is_video"     : int(rec.is_video),
    }
