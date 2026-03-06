import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DatasetConfig, ModelConfig
from src.data.json_loader import load_and_validate_json, build_filename_lookup
from src.data.dataset_builder import build_master_dataframe
from src.models.tfidf_classifier import build_tfidf_features, train_logistic_regression, train_svm, NUMERIC_COLS
from src.models.gbm_classifier import train_lightgbm, train_xgboost
from src.evaluation.evaluator import evaluate_model, find_optimal_threshold

def sample_dataframe(
    df: pd.DataFrame,
    fraction: float,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Stratified-sample the master DataFrame, preserving class balance."""
    if fraction >= 1.0:
        return df
    sampled, _ = train_test_split(
        df, train_size=fraction,
        stratify=df["label"], random_state=random_seed,
    )
    print(f"✅ Using {len(sampled)} / {len(df)} records ({fraction*100:.0f}%)")
    print(sampled["label"].value_counts().rename({1: "Pavbhaji", 0: "Not Pavbhaji"}))
    return sampled.reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="Pavbhaji Post Classifier Pipeline")
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="lightgbm", choices=["logistic_regression", "svm", "lightgbm", "xgboost", "muril"])
    parser.add_argument("--pretrained-model", type=str, default="google/muril-base-cased")
    args = parser.parse_args()

    # 1. Load Data
    print("Loading JSON data...")
    dataset_root = "dataset"
    json_path = f"{dataset_root}/pavbhaji.json"
    
    if not Path(json_path).exists():
        print(f"⚠️ {json_path} not found. Please place data in dataset/. You can create dummy dirs/files for test if dataset empty.")
        return

    dataset_cfg = DatasetConfig(
        dataset_root=dataset_root,
        json_path=json_path,
        sample_fraction=args.sample_fraction
    )
    model_cfg = ModelConfig(
        model_type=args.model,
        pretrained_model_name=args.pretrained_model
    )

    valid_records, failed_records = load_and_validate_json(dataset_cfg.json_path)
    file_lookup = build_filename_lookup(valid_records)

    # 2. Build Dataset
    print("\nBuilding dataset...")
    df = build_master_dataframe(dataset_cfg, file_lookup)
    if df.empty:
        print("Empty dataset.")
        return

    df = sample_dataframe(df, dataset_cfg.sample_fraction, dataset_cfg.random_seed)

    # 3. Train / Val Split
    train_df, val_df = train_test_split(
        df, test_size=dataset_cfg.val_split + dataset_cfg.test_split, 
        stratify=df["label"], random_state=dataset_cfg.random_seed
    )

    # 4. Train Model
    y_train = train_df["label"].tolist()
    y_val = val_df["label"].tolist()
    
    print(f"\nTraining {model_cfg.model_type}...")
    
    if model_cfg.model_type in ("logistic_regression", "svm"):
        X_train_feats, vectorizer = build_tfidf_features(train_df, model_cfg, fit=True)
        X_val_feats, _ = build_tfidf_features(val_df, model_cfg, fit=False, vectorizer=vectorizer)
        
        if model_cfg.model_type == "logistic_regression":
            clf = train_logistic_regression(X_train_feats, y_train)
        else:
            clf = train_svm(X_train_feats, y_train)
        
        y_proba = clf.predict_proba(X_val_feats)[:, 1].tolist()
    
    elif model_cfg.model_type == "lightgbm":
        bst = train_lightgbm(train_df, y_train, val_df, y_val)
        y_proba = bst.predict(val_df[NUMERIC_COLS]).tolist()

    elif model_cfg.model_type == "xgboost":
        import xgboost as xgb
        bst = train_xgboost(train_df, y_train, val_df, y_val)
        dval = xgb.DMatrix(val_df[NUMERIC_COLS])
        y_proba = bst.predict(dval).tolist()
        
    else:
        print(f"Model {model_cfg.model_type} requires transform/trainer script not fully implemented yet in main.py.")
        return

    # 5. Evaluate
    print("\nEvaluating Model...")
    opt_thresh = find_optimal_threshold(y_val, y_proba)
    y_pred = [1 if p >= opt_thresh else 0 for p in y_proba]
    
    evaluate_model(y_val, y_pred, y_proba)

if __name__ == "__main__":
    main()
