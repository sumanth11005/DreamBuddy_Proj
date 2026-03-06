from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    average_precision_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, roc_auc_score,
)


def evaluate_model(
    y_true: List[int],
    y_pred: List[int],
    y_proba: List[float],
    save_dir: str = "outputs/reports/",
) -> Dict[str, float]:
    """Run full evaluation, print all metrics, save plots, return metrics dict."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    report  = classification_report(y_true, y_pred,
                  target_names=["Not Pavbhaji", "Pavbhaji"], output_dict=True)
    auc_roc = roc_auc_score(y_true, y_proba)
    pr_auc  = average_precision_score(y_true, y_proba)
    mcc     = matthews_corrcoef(y_true, y_pred)

    print(classification_report(y_true, y_pred,
          target_names=["Not Pavbhaji", "Pavbhaji"]))
    print(f"AUC-ROC : {auc_roc:.4f}  |  PR-AUC : {pr_auc:.4f}  |  MCC : {mcc:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
        display_labels=["Not Pavbhaji", "Pavbhaji"]).plot(ax=axes[0], cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[1])
    axes[1].set_title(f"ROC Curve (AUC={auc_roc:.3f})")
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=axes[2])
    axes[2].set_title(f"Precision-Recall (AP={pr_auc:.3f})")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/evaluation_plots.png", dpi=150)
    print(f"✅ Plots saved → {save_dir}/evaluation_plots.png")

    return {
        "f1_pavbhaji": report["Pavbhaji"]["f1-score"],
        "precision"  : report["Pavbhaji"]["precision"],
        "recall"     : report["Pavbhaji"]["recall"],
        "auc_roc"    : auc_roc,
        "pr_auc"     : pr_auc,
        "mcc"        : mcc,
    }


def find_optimal_threshold(
    y_true: List[int],
    y_proba: List[float],
) -> float:
    """Grid-search the threshold that maximises Pavbhaji F1 on the validation set."""
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.10, 0.91, 0.05):
        preds = (np.array(y_proba) >= thresh).astype(int)
        score = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, float(thresh)
    print(f"✅ Optimal threshold: {best_thresh:.2f}  (F1 = {best_f1:.4f})")
    return best_thresh
