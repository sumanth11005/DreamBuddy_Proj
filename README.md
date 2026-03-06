#  - TF-IDF + Classical Support Vector Classifier (Under 5 minutes)
uv run python main.py --sample-fraction 1.0 --model svm
#  - Gradient Boosters (LightGBM on Engineered Metrics)
uv run python main.py --sample-fraction 1.0 --model lightgbm
#  - Gradient Boosters (Alternative XGBoost setup)
uv run python main.py --sample-fraction 1.0 --model xgboost
