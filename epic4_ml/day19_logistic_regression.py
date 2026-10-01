"""Day 19: Logistic Regression & Classification Metrics - Epic 4.

Dependency: reuses Day 17's raw Telco loader, feature engineering, and
`build_preprocessor` (with its real NUMERIC_COLS/NOMINAL_COLS/ORDINAL_COLS
for this dataset) - exactly the same preprocessing pipeline Day 17 built
and verified, not a new one-off. The classification dataset is Telco
Customer Churn, same as Day 16/17.

=============================================================================
Why accuracy alone is misleading here (and the whole point of today)
=============================================================================
Day 16's Dummy baseline already showed this: always predicting "no churn"
scores ~73.5% accuracy on this dataset while catching ZERO actual churners.
A real model that also lands around 73-80% accuracy might just be doing the
same lazy thing in disguise. That is exactly why today's metrics look past
accuracy:
    - Precision: of everyone the model FLAGGED as a churn risk, what
      fraction actually churned? Low precision = wasted retention-team
      effort on customers who were never leaving.
    - Recall: of everyone who ACTUALLY churned, what fraction did the
      model catch? Low recall = churners slipping through undetected,
      which is usually the costlier mistake in a churn-prevention program.
    - F1: the harmonic mean of precision and recall - a single number that
      punishes models that sacrifice one entirely for the other.
    - ROC-AUC: how well the model ranks a random churner above a random
      non-churner, across every possible decision threshold at once -
      unlike the other four metrics, it does not depend on picking a
      threshold first.
=============================================================================
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from epic4_ml.day16_ml_basics import get_baseline_scores, split_data
    from epic4_ml.day17_feature_engineering import (
        NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS,
        add_engineered_features, build_preprocessor, load_raw_telco,
    )
except ImportError:
    from day16_ml_basics import get_baseline_scores, split_data
    from day17_feature_engineering import (
        NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS,
        add_engineered_features, build_preprocessor, load_raw_telco,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Matches Day 14's theme exactly, so every chart across Epic 3 and Epic 4
# looks consistent without repeating styling code.
sns.set_theme(style="whitegrid", palette="deep")

CHARTS_DIR = Path(__file__).parent / "charts"
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Sigmoid - numerically stable
# --------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid 1 / (1 + e^-z), stable for large |z|.

    The naive `1 / (1 + np.exp(-z))` overflows for very negative z (e^-z
    becomes astronomically large, `np.exp` warns/returns inf) - not for
    very positive z, since e^-z just underflows to 0 there, which is
    harmless. This implementation picks whichever of two mathematically
    equivalent formulas never exponentiates a large POSITIVE number:
        z >= 0:  1 / (1 + e^-z)       - exponent is <= 0, safe
        z <  0:  e^z / (1 + e^z)      - exponent is < 0, safe
    Both formulas agree exactly at z=0 (both give 0.5) and are algebraically
    identical everywhere else - this is purely a numerical stability fix,
    not a different function.

    Args:
        z: Input array (or scalar) of any real values.

    Returns:
        Array the same shape as `z`, with every value in (0, 1).
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


# --------------------------------------------------------------------------
# 2. Logistic regression - from scratch, gradient descent on log-loss
# --------------------------------------------------------------------------
def logistic_regression_fit(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1000,
) -> Tuple[np.ndarray, List[float]]:
    """Fit logistic regression weights with full-batch gradient descent on log-loss.

    Log-loss (binary cross-entropy): -mean(y*log(p) + (1-y)*log(1-p)), where
    p = sigmoid(X @ w). Its gradient w.r.t. w has the same clean form as
    linear regression's MSE gradient: (1/n) * X^T @ (p - y) - this is not a
    coincidence, it falls out of log-loss being the right loss function to
    pair with the sigmoid (its derivative cancels the sigmoid's own
    derivative in the chain rule).

    Predicted probabilities are clipped away from exactly 0 or 1 before
    taking the log, since log(0) is -inf and would turn one bad prediction
    into an undefined (nan) loss for the whole batch.

    Args:
        X: Feature matrix, shape (n_samples, n_features). Should be scaled
            (e.g. via Day 17's preprocessor) for stable, fast convergence.
        y: Binary target vector (0/1), shape (n_samples,).
        lr: Learning rate.
        epochs: Number of full-batch gradient steps.

    Returns:
        (weights, loss_history) - `weights` shape (n_features + 1,) with
        the intercept first; `loss_history` has length `epochs + 1`.

    Raises:
        ValueError: If `y` contains values other than 0/1, `X`/`y` are
            empty or mismatched, or `lr`/`epochs` are not positive.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] == 0:
        raise ValueError("logistic_regression_fit requires at least one sample")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("y must be binary (contain only 0 and 1) for logistic_regression_fit")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr <= 0:
        raise ValueError(f"lr must be a positive number, got {lr!r}")
    if not isinstance(epochs, (int, np.integer)) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")

    n_samples = X.shape[0]
    X_design = np.hstack([np.ones((n_samples, 1)), X])
    w = np.zeros(X_design.shape[1])
    eps = 1e-12  # clipping margin to keep log() finite

    def log_loss(weights: np.ndarray) -> float:
        p = np.clip(sigmoid(X_design @ weights), eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    loss_history = [log_loss(w)]
    for _ in range(epochs):
        p = sigmoid(X_design @ w)
        gradient = (1.0 / n_samples) * (X_design.T @ (p - y))
        w = w - lr * gradient
        loss_history.append(log_loss(w))

    return w, loss_history


def predict_proba_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict P(y=1) with weights from `logistic_regression_fit` (w[0] = intercept)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return sigmoid(w[0] + X @ w[1:])


def predict_logistic(X: np.ndarray, w: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Predict 0/1 class labels, thresholding `predict_proba_logistic`'s output."""
    return (predict_proba_logistic(X, w) >= threshold).astype(int)


def plot_loss_curve(loss_history: List[float], save_path: str) -> None:
    """Plot logistic regression's log-loss vs. epoch, saved as a .png."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(loss_history)), loss_history, color="#4C72B0", linewidth=2)
    ax.set_title("Logistic Regression Loss Curve (Log-Loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log-Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved loss curve: %s", save_path)


# --------------------------------------------------------------------------
# 3. Classification metrics
# --------------------------------------------------------------------------
def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Compute confusion matrix, accuracy, precision, recall, F1, and ROC-AUC.

    - Confusion matrix layout (sklearn's default, class 0 = no churn first):
          [[TN, FP],
           [FN, TP]]
      FN (false negative = an actual churner predicted "no churn") is
      usually the costlier mistake in a churn-prevention program, since it
      means losing a customer with no retention attempt at all.
    - Precision, recall, F1 are computed for the POSITIVE class (y=1,
      "churn") by default - the class usually of actual business interest
      in an imbalanced problem like this one.
    - ROC-AUC uses `y_prob` (continuous probabilities), not `y_pred` -
      it measures ranking quality independent of any one threshold.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred: Predicted binary labels (0/1), at whatever threshold was used.
        y_prob: Predicted probabilities of the positive class, in [0, 1].

    Returns:
        Dict with keys: 'confusion_matrix' (2x2 ndarray), 'accuracy',
        'precision', 'recall', 'f1', 'roc_auc'.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_prob = np.asarray(y_prob).ravel()

    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray, thresholds: List[float]) -> pd.DataFrame:
    """Compute precision/recall/F1 at each of several decision thresholds.

    Raising the threshold makes the model more conservative about
    predicting "churn" - precision tends to go up (fewer false alarms
    among positive predictions) while recall tends to go down (more actual
    churners missed). This table makes that trade-off visible numerically
    instead of hand-wavy.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_prob: Predicted probabilities of the positive class.
        thresholds: Decision thresholds to evaluate, e.g. [0.3, 0.5, 0.7].

    Returns:
        DataFrame with one row per threshold: columns 'threshold',
        'precision', 'recall', 'f1'.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Plots
# --------------------------------------------------------------------------
def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str, label: str = "Model") -> None:
    """Plot the ROC curve (true positive rate vs. false positive rate) with its AUC, saved as a .png."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random guess (AUC = 0.5)")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve: %s", save_path)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray, save_path: str, labels: Tuple[str, str] = ("No churn", "Churn"),
) -> None:
    """Plot a confusion matrix as an annotated Seaborn heatmap, saved as a .png."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix heatmap: %s", save_path)


# --------------------------------------------------------------------------
# 5. Demonstration
# --------------------------------------------------------------------------
def _model_row(name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": name,
        "accuracy": round(report["accuracy"], 4),
        "precision": round(report["precision"], 4),
        "recall": round(report["recall"], 4),
        "f1": round(report["f1"], 4),
        "roc_auc": round(report["roc_auc"], 4),
    }


def main() -> None:
    logger.info("=== Day 19: Logistic Regression & Classification Metrics ===")

    raw = load_raw_telco()
    y = raw["Churn"]
    X_raw = raw.drop(columns=["Churn"])
    X_train_raw, X_test_raw, y_train, y_test = split_data(X_raw, y, test_size=0.2, stratify=True, random_state=RANDOM_STATE)

    X_train_fe = add_engineered_features(X_train_raw)
    X_test_fe = add_engineered_features(X_test_raw)

    preprocessor = build_preprocessor(NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS)
    X_train_processed = preprocessor.fit_transform(X_train_fe, y_train)
    X_test_processed = preprocessor.transform(X_test_fe)
    logger.info("Preprocessed shapes: train %s, test %s", X_train_processed.shape, X_test_processed.shape)

    # --- Day 16 baseline, for reference ---
    baseline = get_baseline_scores(X_train_processed, X_test_processed, y_train, y_test, task="classification")
    logger.info("Day 16 baseline (always predict majority class): accuracy=%.4f, f1=%.4f",
                baseline["test_accuracy"], baseline["test_f1"])

    # --- From-scratch logistic regression ---
    y_train_arr = y_train.to_numpy()
    w_scratch, loss_history = logistic_regression_fit(X_train_processed, y_train_arr, lr=0.3, epochs=2000)
    plot_loss_curve(loss_history, str(CHARTS_DIR / "day19_logistic_loss.png"))
    logger.info("From-scratch logistic regression: log-loss %.4f -> %.4f over %d epochs",
                loss_history[0], loss_history[-1], len(loss_history) - 1)

    y_prob_scratch = predict_proba_logistic(X_test_processed, w_scratch)
    y_pred_scratch = predict_logistic(X_test_processed, w_scratch, threshold=0.5)

    # --- sklearn LogisticRegression, inside an actual Pipeline with the Day 17 preprocessor ---
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS)),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]).fit(X_train_fe, y_train)
    y_prob_sklearn = pipeline.predict_proba(X_test_fe)[:, 1]
    y_pred_sklearn = pipeline.predict(X_test_fe)

    agreement = float(np.mean(y_pred_scratch == y_pred_sklearn))
    logger.info("From-scratch vs. sklearn prediction agreement: %.1f%% of test samples", agreement * 100)

    # --- Balanced class weights, to handle the ~73.5%/26.5% imbalance ---
    balanced_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    balanced_model.fit(X_train_processed, y_train)
    y_prob_balanced = balanced_model.predict_proba(X_test_processed)[:, 1]
    y_pred_balanced = balanced_model.predict(X_test_processed)

    # --- Metrics for all four ---
    report_scratch = classification_report_dict(y_test, y_pred_scratch, y_prob_scratch)
    report_sklearn = classification_report_dict(y_test, y_pred_sklearn, y_prob_sklearn)
    report_balanced = classification_report_dict(y_test, y_pred_balanced, y_prob_balanced)

    logger.info("--- Model comparison ---")
    comparison = pd.DataFrame([
        {"model": "Baseline (Dummy, most_frequent)", "accuracy": round(baseline["test_accuracy"], 4),
         "precision": 0.0, "recall": 0.0, "f1": round(baseline["test_f1"], 4), "roc_auc": 0.5},
        _model_row("From-scratch logistic regression", report_scratch),
        _model_row("sklearn LogisticRegression", report_sklearn),
        _model_row("sklearn LogisticRegression (balanced)", report_balanced),
    ])
    logger.info("\n%s", comparison.to_string(index=False))

    plot_roc_curve(y_test, y_prob_sklearn, str(CHARTS_DIR / "day19_roc_curve.png"), label="LogisticRegression")
    plot_confusion_matrix_heatmap(report_sklearn["confusion_matrix"], str(CHARTS_DIR / "day19_confusion_matrix.png"))

    # --- Threshold analysis ---
    logger.info("--- Threshold analysis (sklearn LogisticRegression probabilities) ---")
    threshold_table = threshold_analysis(y_test, y_prob_sklearn, thresholds=[0.3, 0.5, 0.7])
    logger.info("\n%s", threshold_table.to_string(index=False))
    logger.info(
        "Business pick for churn PREVENTION: threshold=0.3. Missing an actual churner (false negative) costs a "
        "lost customer entirely, while a false alarm only costs one unnecessary retention outreach - so recall "
        "matters more than precision here, and 0.3 catches more real churners at an acceptable precision cost."
    )


if __name__ == "__main__":
    main()
