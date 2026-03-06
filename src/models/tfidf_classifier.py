from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.config import ModelConfig

NUMERIC_COLS = [
    "caption_length", "word_count", "mentions_pavbhaji", "mentions_butter",
    "mentions_recipe", "has_hindi_text", "emoji_food_count", "exclamation_count",
    "has_pavbhaji_tag", "has_confusion_tag", "has_mumbai_tag", "has_food_tag",
    "total_tags", "pavbhaji_tag_ratio", "log_likes", "log_comments",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "has_location", "aspect_ratio", "is_square", "json_is_video",
]


def build_tfidf_features(
    df: pd.DataFrame,
    config: ModelConfig,
    fit: bool = True,
    vectorizer: TfidfVectorizer = None,
) -> Tuple[object, TfidfVectorizer]:
    """Fit (or transform) TF-IDF on caption_raw, optionally stack numeric features."""
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=(config.tfidf_ngram_min, config.tfidf_ngram_max),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )
        text_mat = vectorizer.fit_transform(df["caption_raw"].fillna(""))
    else:
        text_mat = vectorizer.transform(df["caption_raw"].fillna(""))

    if config.use_engineered_features:
        numeric  = df[NUMERIC_COLS].fillna(0).values.astype(np.float32)
        features = hstack([text_mat, csr_matrix(numeric)])
    else:
        features = text_mat

    return features, vectorizer


def train_logistic_regression(X_train, y_train: List[int]) -> LogisticRegression:
    """Fit a class-balanced Logistic Regression classifier."""
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X_train, y_train)
    return clf


def train_svm(X_train, y_train: List[int]) -> SVC:
    """Fit a class-balanced SVM classifier with linear kernel."""
    clf = SVC(kernel="linear", class_weight="balanced", probability=True, C=1.0)
    clf.fit(X_train, y_train)
    return clf
