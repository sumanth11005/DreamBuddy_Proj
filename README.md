# Tier 2 - TF-IDF + Classical Support Vector Classifier (Under 5 minutes)
uv run python main.py --sample-fraction 1.0 --model svm
# Tier 2 - Gradient Boosters (LightGBM on Engineered Metrics)
uv run python main.py --sample-fraction 1.0 --model lightgbm
# Tier 2 - Gradient Boosters (Alternative XGBoost setup)
uv run python main.py --sample-fraction 1.0 --model xgboost