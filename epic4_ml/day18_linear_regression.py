"""Day 18: Linear Regression & Regression Metrics - Epic 4.

Dependency: reuses Day 17's `build_preprocessor` (with empty categorical/
ordinal column lists, since the Diabetes dataset - Day 16's regression
dataset - is entirely numeric) and Day 16's `split_data`/`load_ml_dataset`.
The preprocessor guarantees every model here (from-scratch and sklearn
alike) trains on the exact same median-imputed, standardized feature
matrix, which is what makes the "from-scratch coefficients must match
sklearn's" comparison a fair one.

=============================================================================
Why compare against a Pipeline, not raw sklearn.LinearRegression alone?
=============================================================================
`normal_equation_fit` and `gradient_descent_fit` both operate on whatever
NumPy array they are handed - they have no idea a preprocessing step ever
happened. So both `sklearn.LinearRegression` AND the from-scratch functions
are fit on IDENTICAL preprocessed arrays (`X_train_processed`) in `main()`.
If the from-scratch functions were instead compared against a model fit on
raw, unscaled data, any mismatch would be ambiguous - is it a bug, or just
a different (equally valid) coordinate system? Fitting everything on the
same processed matrix removes that ambiguity entirely.
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
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from epic4_ml.day16_ml_basics import load_ml_dataset, split_data
    from epic4_ml.day17_feature_engineering import build_preprocessor, get_feature_names
except ImportError:
    from day16_ml_basics import load_ml_dataset, split_data
    from day17_feature_engineering import build_preprocessor, get_feature_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHARTS_DIR = Path(__file__).parent / "charts"
REGRESSION_DATASET = "diabetes"
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Normal equation - closed-form solution
# --------------------------------------------------------------------------
def normal_equation_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit linear regression weights with the normal equation: w = (X^T X)^-1 X^T y.

    A bias/intercept column of 1s is prepended to `X` internally, so the
    caller passes plain features and gets back `w` where `w[0]` is the
    intercept and `w[1:]` line up with `X`'s columns in order.

    Singular matrix handling: (X^T X) is singular (non-invertible) when its
    columns are linearly dependent - e.g. two duplicated/perfectly
    collinear features, a constant (zero-variance) column, or simply more
    features than samples (n_features > n_samples, so there are infinitely
    many exact-fit solutions). `np.linalg.solve` raises `LinAlgError` in
    exactly that case; the fallback uses `np.linalg.pinv` (the Moore-Penrose
    pseudo-inverse), which always exists and returns the minimum-norm
    solution among the infinitely many that fit equally well - a sane,
    deterministic choice rather than crashing.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        y: Target vector, shape (n_samples,).

    Returns:
        Weight vector `w`, shape (n_features + 1,), intercept first.

    Raises:
        ValueError: If `X` and `y` have mismatched sample counts, or `X`
            is empty.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] == 0:
        raise ValueError("normal_equation_fit requires at least one sample")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}")

    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    XtX = X_design.T @ X_design
    Xty = X_design.T @ y

    try:
        w = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        logger.warning(
            "X^T X is singular (collinear/duplicate features, a constant column, or "
            "n_features > n_samples) - falling back to the Moore-Penrose pseudo-inverse."
        )
        w = np.linalg.pinv(XtX) @ Xty

    return w


def predict_linear(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict with weights from `normal_equation_fit`/`gradient_descent_fit` (w[0] = intercept)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return w[0] + X @ w[1:]


# --------------------------------------------------------------------------
# 2. Gradient descent
# --------------------------------------------------------------------------
def gradient_descent_fit(
    X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 500,
) -> Tuple[np.ndarray, List[float]]:
    """Fit linear regression weights with full-batch gradient descent on the MSE loss.

    Full-batch (uses every sample each step, not a random mini-batch) means
    the loss is mathematically guaranteed to be non-increasing step to step
    for a small enough `lr`, since MSE is a convex quadratic in `w` - no
    randomness to cause a step backward. A `lr` that is too large for the
    feature scale can still diverge (loss increases or explodes), which is
    exactly why Day 17's standardized features matter here: gradient
    descent on RAW, unscaled columns of very different magnitudes needs a
    tiny, hard-to-guess `lr` per column, while standardized features (mean
    0, std 1) behave consistently regardless of the original units.

    Args:
        X: Feature matrix, shape (n_samples, n_features). Should be scaled
            (e.g. via Day 17's preprocessor) for stable convergence.
        y: Target vector, shape (n_samples,).
        lr: Learning rate (step size).
        epochs: Number of full-batch gradient steps.

    Returns:
        (weights, loss_history) - `weights` shape (n_features + 1,) with
        the intercept first, and `loss_history` a list of length
        `epochs + 1` (the initial loss, before any update, then one entry
        per epoch).

    Raises:
        ValueError: If `lr` is not positive, `epochs` is not a positive
            integer, or `X`/`y` are mismatched/empty.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] == 0:
        raise ValueError("gradient_descent_fit requires at least one sample")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr <= 0:
        raise ValueError(f"lr must be a positive number, got {lr!r}")
    if not isinstance(epochs, (int, np.integer)) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")

    n_samples = X.shape[0]
    X_design = np.hstack([np.ones((n_samples, 1)), X])
    w = np.zeros(X_design.shape[1])

    def mse_loss(weights: np.ndarray) -> float:
        residuals = X_design @ weights - y
        return float(np.mean(residuals ** 2))

    loss_history = [mse_loss(w)]
    for _ in range(epochs):
        residuals = X_design @ w - y
        gradient = (2.0 / n_samples) * (X_design.T @ residuals)
        w = w - lr * gradient
        loss_history.append(mse_loss(w))

    return w, loss_history


def plot_loss_curve(loss_history: List[float], save_path: str) -> None:
    """Plot gradient descent's loss vs. epoch, saved as a .png.

    A healthy curve drops steeply at first then flattens out (diminishing
    returns as w approaches the optimum). A curve that goes back UP at any
    point means `lr` was too large for the feature scale.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(loss_history)), loss_history, color="#4C72B0", linewidth=2)
    ax.set_title("Gradient Descent Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved loss curve: %s", save_path)


# --------------------------------------------------------------------------
# 3. Regression metrics - from scratch
# --------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, and R^2 from scratch (no sklearn.metrics calls).

    - MAE  (Mean Absolute Error): average of |y_true - y_pred|. Same units
      as the target; treats every error linearly (a $20 error counts twice
      a $10 error).
    - MSE  (Mean Squared Error): average of (y_true - y_pred)^2. Penalizes
      large errors disproportionately (a $20 error counts 4x a $10 error).
    - RMSE (Root MSE): sqrt(MSE). Brings the units back to match the
      target (unlike MSE, which is in squared units), while keeping MSE's
      sensitivity to large errors.
    - R^2  (coefficient of determination): 1 - SS_res / SS_tot, where
      SS_res = sum((y_true - y_pred)^2) and SS_tot = sum((y_true - mean(y_true))^2).
      The fraction of the target's variance the model explains: 1.0 is a
      perfect fit, 0.0 matches predicting the mean every time (the Day 16
      Dummy baseline), and it can go negative if the model is worse than
      that baseline.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values (same length as `y_true`).

    Returns:
        Dict with keys 'mae', 'mse', 'rmse', 'r2'.

    Raises:
        ValueError: If `y_true` and `y_pred` have different lengths, or
            either is empty.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape[0] == 0:
        raise ValueError("regression_metrics requires at least one value")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred must be the same length, got {y_true.shape[0]} and {y_pred.shape[0]}")

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# --------------------------------------------------------------------------
# 4. Residual diagnostics
# --------------------------------------------------------------------------
def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Plot predicted-vs-actual and residuals-vs-predicted, side by side, saved as a .png.

    - Left (predicted vs. actual): points hugging the diagonal y=x line
      mean accurate predictions; a systematic curve or fan shape flags a
      pattern the model is missing.
    - Right (residuals vs. predicted): if the model's assumptions hold,
      residuals should scatter randomly around the zero line with roughly
      constant spread. A funnel/cone shape (spread growing with the
      predicted value) signals heteroscedasticity - the model's errors are
      not uniform, so its uncertainty estimates would be misleading.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.
        save_path: Where to save the combined .png.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolor="black", color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, color="crimson", linestyle="--", linewidth=1.5, label="Perfect prediction (y = x)")
    axes[0].set_title("Predicted vs. Actual")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolor="black", color="#C44E52")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs. Predicted")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual (Actual - Predicted)")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved residual plot: %s", save_path)


# --------------------------------------------------------------------------
# 5. Demonstration
# --------------------------------------------------------------------------
def _model_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    m = regression_metrics(y_true, y_pred)
    return {"model": name, "MAE": round(m["mae"], 3), "RMSE": round(m["rmse"], 3), "R2": round(m["r2"], 4)}


def main() -> None:
    logger.info("=== Day 18: Linear Regression & Regression Metrics ===")

    X, y = load_ml_dataset(REGRESSION_DATASET)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Reuse Day 17's preprocessor with empty categorical/ordinal lists -
    # Diabetes is entirely numeric, so only the median-impute + scale half
    # of the pipeline applies.
    preprocessor = build_preprocessor(numeric_cols=list(X.columns), categorical_cols=[], ordinal_cols={})
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    logger.info("Preprocessed shapes: train %s, test %s", X_train_processed.shape, X_test_processed.shape)

    # --- Normal equation vs. sklearn LinearRegression, on the SAME processed matrix ---
    w_normal_eq = normal_equation_fit(X_train_processed, y_train)
    y_pred_normal_eq = predict_linear(X_test_processed, w_normal_eq)

    sklearn_lr = LinearRegression().fit(X_train_processed, y_train)
    y_pred_sklearn = sklearn_lr.predict(X_test_processed)

    coef_diff = np.abs(w_normal_eq[1:] - sklearn_lr.coef_)
    intercept_diff = abs(w_normal_eq[0] - sklearn_lr.intercept_)
    logger.info(
        "Normal equation vs. sklearn LinearRegression: max |coefficient difference| = %.2e, "
        "|intercept difference| = %.2e", coef_diff.max(), intercept_diff,
    )

    # --- Gradient descent, on the same processed matrix ---
    w_gd, loss_history = gradient_descent_fit(X_train_processed, y_train, lr=0.05, epochs=1000)
    y_pred_gd = predict_linear(X_test_processed, w_gd)
    plot_loss_curve(loss_history, str(CHARTS_DIR / "day18_gradient_descent_loss.png"))
    logger.info("Gradient descent: loss %.2f -> %.2f over %d epochs", loss_history[0], loss_history[-1], len(loss_history) - 1)

    # --- Metrics: from-scratch vs. sklearn.metrics, cross-validated ---
    scratch_metrics = regression_metrics(y_test, y_pred_sklearn)
    sklearn_metrics = {
        "mae": mean_absolute_error(y_test, y_pred_sklearn),
        "mse": mean_squared_error(y_test, y_pred_sklearn),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_sklearn))),
        "r2": r2_score(y_test, y_pred_sklearn),
    }
    logger.info("Metric cross-validation (from-scratch vs sklearn.metrics): scratch=%s sklearn=%s",
                {k: round(v, 6) for k, v in scratch_metrics.items()},
                {k: round(v, 6) for k, v in sklearn_metrics.items()})

    plot_residuals(y_test, y_pred_sklearn, str(CHARTS_DIR / "day18_residuals.png"))

    # --- Ridge & Lasso, same processed features - see regularization's effect on coefficients ---
    ridge = Ridge(alpha=1.0).fit(X_train_processed, y_train)
    lasso = Lasso(alpha=0.5).fit(X_train_processed, y_train)
    y_pred_ridge = ridge.predict(X_test_processed)
    y_pred_lasso = lasso.predict(X_test_processed)

    logger.info("--- Coefficient comparison (regularization shrinks/zeros coefficients) ---")
    coef_table = pd.DataFrame({
        "feature": feature_names,
        "LinearRegression": sklearn_lr.coef_.round(3),
        "Ridge(alpha=1.0)": ridge.coef_.round(3),
        "Lasso(alpha=0.5)": lasso.coef_.round(3),
    })
    logger.info("\n%s", coef_table.to_string(index=False))
    logger.info("Lasso zeroed out %d of %d coefficients (automatic feature selection); "
                "Ridge shrank all coefficients toward zero but kept them nonzero.",
                int((lasso.coef_ == 0).sum()), len(lasso.coef_))

    # --- Results table ---
    results = pd.DataFrame([
        _model_row("From-scratch (normal equation)", y_test, y_pred_normal_eq),
        _model_row("From-scratch (gradient descent)", y_test, y_pred_gd),
        _model_row("sklearn LinearRegression", y_test, y_pred_sklearn),
        _model_row("Ridge (alpha=1.0)", y_test, y_pred_ridge),
        _model_row("Lasso (alpha=0.5)", y_test, y_pred_lasso),
    ])
    logger.info("--- Model comparison ---\n%s", results.to_string(index=False))

    # --- Singular matrix edge case demonstration ---
    X_singular = np.column_stack([X_train_processed[:, 0], X_train_processed[:, 0]])  # duplicated column
    w_singular = normal_equation_fit(X_singular, y_train)
    logger.info("Singular-matrix case (duplicated column) handled without crashing: weights=%s", np.round(w_singular, 3))


if __name__ == "__main__":
    main()
