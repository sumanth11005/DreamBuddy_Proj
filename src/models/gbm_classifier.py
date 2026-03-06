from typing import List

import lightgbm as lgb
import xgboost as xgb
import pandas as pd

from src.models.tfidf_classifier import NUMERIC_COLS


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: List[int],
    X_val: pd.DataFrame,
    y_val: List[int],
) -> lgb.Booster:
    """Train a LightGBM binary classifier on engineered features."""
    neg = sum(1 for y in y_train if y == 0)
    pos = sum(1 for y in y_train if y == 1)

    params = {
        "objective"        : "binary",
        "metric"           : "binary_logloss",
        "scale_pos_weight" : neg / max(1, pos),
        "learning_rate"    : 0.05,
        "num_leaves"       : 63,
        "min_child_samples": 20,
        "feature_fraction" : 0.8,
        "bagging_fraction" : 0.8,
        "bagging_freq"     : 5,
        "verbosity"        : -1,
    }

    train_data = lgb.Dataset(X_train[NUMERIC_COLS], label=y_train)
    val_data   = lgb.Dataset(X_val[NUMERIC_COLS],   label=y_val, reference=train_data)

    return lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: List[int],
    X_val: pd.DataFrame,
    y_val: List[int],
) -> xgb.Booster:
    """Train an XGBoost binary classifier on engineered features."""
    neg = sum(1 for y in y_train if y == 0)
    pos = sum(1 for y in y_train if y == 1)

    dtrain = xgb.DMatrix(X_train[NUMERIC_COLS], label=y_train)
    dval   = xgb.DMatrix(X_val[NUMERIC_COLS], label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": neg / max(1, pos),
        "learning_rate": 0.05,
        "max_depth": 6,
    }

    return xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50
    )
