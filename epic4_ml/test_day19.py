"""Pytest suite for day19_logistic_regression.py.

Run with: pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

file_dir = Path(__file__).parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic4_ml.day19_logistic_regression import (
        classification_report_dict,
        logistic_regression_fit,
        plot_confusion_matrix_heatmap,
        plot_roc_curve,
        predict_logistic,
        predict_proba_logistic,
        sigmoid,
        threshold_analysis,
    )
except ImportError:
    from day19_logistic_regression import (
        classification_report_dict,
        logistic_regression_fit,
        plot_confusion_matrix_heatmap,
        plot_roc_curve,
        predict_logistic,
        predict_proba_logistic,
        sigmoid,
        threshold_analysis,
    )


@pytest.fixture
def separable_synthetic():
    """A cleanly (but not perfectly) separable binary classification problem."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(0, 1, size=(n, 3))
    true_w = np.array([2.0, -1.5, 0.5])
    z = X @ true_w + 0.5
    prob = 1 / (1 + np.exp(-z))
    y = (rng.uniform(size=n) < prob).astype(int)
    return X, y


# --------------------------------------------------------------------------
# sigmoid
# --------------------------------------------------------------------------
def test_sigmoid_at_zero_is_half():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_stable_for_large_magnitude_inputs():
    # Overflow/invalid (nan) would mean the function is broken; underflow to
    # exact 0.0 for np.exp of a very negative number is correct, harmless
    # float64 behavior (sigmoid genuinely IS 1.0 or 0.0 at this magnitude),
    # so only overflow/invalid are treated as failures here.
    with np.errstate(over="raise", invalid="raise"):
        result = sigmoid(np.array([1000.0, -1000.0, 10000.0, -10000.0]))
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(1.0, abs=1e-9)
    assert result[1] == pytest.approx(0.0, abs=1e-9)
    assert result[2] == pytest.approx(1.0, abs=1e-9)
    assert result[3] == pytest.approx(0.0, abs=1e-9)


def test_sigmoid_output_always_in_zero_one():
    # Inclusive bounds: for |z| large enough (~ > 37), sigmoid's true value
    # is so close to 0 or 1 that float64 correctly rounds it to exactly
    # 0.0/1.0 - that is accurate saturation, not an out-of-range bug.
    values = np.linspace(-50, 50, 1000)
    result = sigmoid(values)
    assert np.all((result >= 0) & (result <= 1))


def test_sigmoid_is_monotonically_increasing():
    values = np.sort(np.random.default_rng(1).uniform(-20, 20, size=100))
    result = sigmoid(values)
    assert np.all(np.diff(result) >= 0)


# --------------------------------------------------------------------------
# logistic_regression_fit / predict_proba_logistic / predict_logistic
# --------------------------------------------------------------------------
def test_logistic_regression_predictions_agree_with_sklearn(separable_synthetic):
    X, y = separable_synthetic
    w, _ = logistic_regression_fit(X, y, lr=0.3, epochs=3000)
    y_pred_scratch = predict_logistic(X, w)

    sklearn_model = LogisticRegression(max_iter=1000).fit(X, y)
    y_pred_sklearn = sklearn_model.predict(X)

    agreement = np.mean(y_pred_scratch == y_pred_sklearn)
    assert agreement >= 0.90


def test_logistic_regression_loss_decreases(separable_synthetic):
    X, y = separable_synthetic
    _, loss_history = logistic_regression_fit(X, y, lr=0.3, epochs=500)
    assert loss_history[-1] < loss_history[0]
    # Loss should trend down overall even if not perfectly monotonic epoch-to-epoch at this lr.
    assert loss_history[-1] < loss_history[len(loss_history) // 2]


def test_predict_proba_in_zero_one_range(separable_synthetic):
    X, y = separable_synthetic
    w, _ = logistic_regression_fit(X, y, lr=0.3, epochs=500)
    proba = predict_proba_logistic(X, w)
    assert np.all((proba >= 0) & (proba <= 1))


def test_logistic_regression_rejects_non_binary_target():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 2])  # not binary
    with pytest.raises(ValueError):
        logistic_regression_fit(X, y, lr=0.1, epochs=10)


def test_logistic_regression_invalid_lr_raises():
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        logistic_regression_fit(X, y, lr=0, epochs=10)


def test_logistic_regression_invalid_epochs_raises():
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        logistic_regression_fit(X, y, lr=0.1, epochs=0)


def test_logistic_regression_mismatched_lengths_raises():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        logistic_regression_fit(X, y, lr=0.1, epochs=10)


# --------------------------------------------------------------------------
# classification_report_dict
# --------------------------------------------------------------------------
def test_classification_report_metrics_in_zero_one_range(separable_synthetic):
    X, y = separable_synthetic
    w, _ = logistic_regression_fit(X, y, lr=0.3, epochs=1000)
    y_prob = predict_proba_logistic(X, w)
    y_pred = predict_logistic(X, w)

    report = classification_report_dict(y, y_pred, y_prob)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= report[key] <= 1.0, f"{key} out of [0, 1]: {report[key]}"


def test_classification_report_confusion_matrix_shape_and_total(separable_synthetic):
    X, y = separable_synthetic
    w, _ = logistic_regression_fit(X, y, lr=0.3, epochs=1000)
    y_pred = predict_logistic(X, w)
    report = classification_report_dict(y, y_pred, predict_proba_logistic(X, w))

    cm = report["confusion_matrix"]
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y)


def test_classification_report_perfect_predictions():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = y_true.copy()
    y_prob = y_true.astype(float)
    report = classification_report_dict(y_true, y_pred, y_prob)
    assert report["accuracy"] == 1.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["f1"] == 1.0


# --------------------------------------------------------------------------
# threshold_analysis
# --------------------------------------------------------------------------
def test_threshold_analysis_returns_one_row_per_threshold():
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, size=100)
    y_prob = rng.uniform(size=100)
    thresholds = [0.3, 0.5, 0.7]

    result = threshold_analysis(y_true, y_prob, thresholds)
    assert len(result) == len(thresholds)
    assert list(result["threshold"]) == thresholds
    assert {"precision", "recall", "f1"} <= set(result.columns)


def test_threshold_analysis_higher_threshold_lowers_recall():
    # Construct probabilities where raising the threshold strictly reduces
    # how many positives get predicted, hence recall can only go down (or
    # stay flat), never up.
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.6, 0.2, 0.85, 0.4, 0.3, 0.1])

    result = threshold_analysis(y_true, y_prob, [0.3, 0.5, 0.7])
    recalls = result.sort_values("threshold")["recall"].tolist()
    assert recalls[0] >= recalls[1] >= recalls[2]


def test_threshold_analysis_values_in_zero_one_range():
    rng = np.random.default_rng(6)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(size=200)
    result = threshold_analysis(y_true, y_prob, [0.2, 0.5, 0.8])
    for col in ("precision", "recall", "f1"):
        assert result[col].between(0, 1).all()


# --------------------------------------------------------------------------
# Plotting (just needs to produce valid files without raising)
# --------------------------------------------------------------------------
def test_plot_roc_curve_creates_valid_png(tmp_path):
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=100)
    y_prob = rng.uniform(size=100)
    out = tmp_path / "roc.png"
    plot_roc_curve(y_true, y_prob, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_confusion_matrix_heatmap_creates_valid_png(tmp_path):
    cm = np.array([[50, 10], [5, 35]])
    out = tmp_path / "cm.png"
    plot_confusion_matrix_heatmap(cm, str(out))
    assert out.exists() and out.stat().st_size > 0
