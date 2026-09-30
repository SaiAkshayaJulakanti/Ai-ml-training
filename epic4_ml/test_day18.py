"""Pytest suite for day18_linear_regression.py.

Run with: pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_dir = Path(__file__).parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic4_ml.day18_linear_regression import (
        gradient_descent_fit,
        normal_equation_fit,
        plot_residuals,
        predict_linear,
        regression_metrics,
    )
except ImportError:
    from day18_linear_regression import (
        gradient_descent_fit,
        normal_equation_fit,
        plot_residuals,
        predict_linear,
        regression_metrics,
    )

TOLERANCE = 1e-6


@pytest.fixture
def linear_synthetic():
    """y = 3x + 2 plus small noise - the exact case the task brief names."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(200, 1))
    noise = rng.normal(0, 0.1, size=200)
    y = (3 * X.ravel() + 2) + noise
    return X, y


@pytest.fixture
def multivariate_synthetic():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(300, 4))
    true_w = np.array([1.5, -2.0, 0.5, 3.0])
    y = X @ true_w + 4.0 + rng.normal(0, 0.05, size=300)
    return X, y


# --------------------------------------------------------------------------
# normal_equation_fit
# --------------------------------------------------------------------------
def test_normal_equation_matches_sklearn_on_linear_synthetic(linear_synthetic):
    X, y = linear_synthetic
    w = normal_equation_fit(X, y)
    sklearn_model = LinearRegression().fit(X, y)

    assert w[0] == pytest.approx(sklearn_model.intercept_, abs=1e-4)
    assert w[1:] == pytest.approx(sklearn_model.coef_, abs=1e-4)


def test_normal_equation_recovers_known_coefficients(linear_synthetic):
    X, y = linear_synthetic
    w = normal_equation_fit(X, y)
    assert w[0] == pytest.approx(2.0, abs=0.1)   # intercept
    assert w[1] == pytest.approx(3.0, abs=0.1)   # slope


def test_normal_equation_matches_sklearn_multivariate(multivariate_synthetic):
    X, y = multivariate_synthetic
    w = normal_equation_fit(X, y)
    sklearn_model = LinearRegression().fit(X, y)
    assert w[0] == pytest.approx(sklearn_model.intercept_, abs=1e-4)
    assert w[1:] == pytest.approx(sklearn_model.coef_, abs=1e-4)


def test_normal_equation_handles_singular_matrix_without_crashing():
    # A duplicated column makes X^T X exactly singular.
    rng = np.random.default_rng(2)
    col = rng.normal(size=100)
    X = np.column_stack([col, col])  # perfectly collinear
    y = 2 * col + 1

    w = normal_equation_fit(X, y)  # must not raise
    assert np.all(np.isfinite(w))
    # The two (identical) columns should split the true slope (2.0) between them.
    assert (w[1] + w[2]) == pytest.approx(2.0, abs=0.1)


def test_normal_equation_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        normal_equation_fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0, 3.0]))


def test_normal_equation_raises_on_empty_data():
    with pytest.raises(ValueError):
        normal_equation_fit(np.empty((0, 2)), np.array([]))


# --------------------------------------------------------------------------
# gradient_descent_fit
# --------------------------------------------------------------------------
def test_gradient_descent_loss_decreases_monotonically(linear_synthetic):
    X, y = linear_synthetic
    _, loss_history = gradient_descent_fit(X, y, lr=0.05, epochs=200)
    diffs = np.diff(loss_history)
    assert np.all(diffs <= 1e-8)  # non-increasing (allowing for float noise)


def test_gradient_descent_converges_close_to_normal_equation(linear_synthetic):
    X, y = linear_synthetic
    w_gd, _ = gradient_descent_fit(X, y, lr=0.1, epochs=2000)
    w_normal_eq = normal_equation_fit(X, y)
    assert w_gd == pytest.approx(w_normal_eq, abs=0.05)


def test_gradient_descent_invalid_lr_raises(linear_synthetic):
    X, y = linear_synthetic
    with pytest.raises(ValueError):
        gradient_descent_fit(X, y, lr=0, epochs=10)
    with pytest.raises(ValueError):
        gradient_descent_fit(X, y, lr=-0.1, epochs=10)


def test_gradient_descent_invalid_epochs_raises(linear_synthetic):
    X, y = linear_synthetic
    with pytest.raises(ValueError):
        gradient_descent_fit(X, y, lr=0.05, epochs=0)


def test_gradient_descent_loss_history_length(linear_synthetic):
    X, y = linear_synthetic
    _, loss_history = gradient_descent_fit(X, y, lr=0.05, epochs=50)
    assert len(loss_history) == 51  # initial loss + 50 epochs


# --------------------------------------------------------------------------
# predict_linear
# --------------------------------------------------------------------------
def test_predict_linear_matches_manual_computation():
    w = np.array([1.0, 2.0, -1.0])
    X = np.array([[1.0, 1.0], [2.0, 0.0]])
    predictions = predict_linear(X, w)
    expected = np.array([1.0 + 2.0 * 1.0 - 1.0 * 1.0, 1.0 + 2.0 * 2.0 - 1.0 * 0.0])
    assert predictions == pytest.approx(expected)


# --------------------------------------------------------------------------
# regression_metrics
# --------------------------------------------------------------------------
def test_regression_metrics_match_sklearn(multivariate_synthetic):
    X, y = multivariate_synthetic
    y_pred = LinearRegression().fit(X, y).predict(X)

    scratch = regression_metrics(y, y_pred)
    assert scratch["mae"] == pytest.approx(mean_absolute_error(y, y_pred), abs=TOLERANCE)
    assert scratch["mse"] == pytest.approx(mean_squared_error(y, y_pred), abs=TOLERANCE)
    assert scratch["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y, y_pred)), abs=TOLERANCE)
    assert scratch["r2"] == pytest.approx(r2_score(y, y_pred), abs=TOLERANCE)


def test_regression_metrics_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(y, y.copy())
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == pytest.approx(1.0)


def test_regression_metrics_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))


def test_regression_metrics_raises_on_empty_input():
    with pytest.raises(ValueError):
        regression_metrics(np.array([]), np.array([]))


# --------------------------------------------------------------------------
# plot_residuals
# --------------------------------------------------------------------------
def test_plot_residuals_creates_valid_png(tmp_path):
    rng = np.random.default_rng(3)
    y_true = rng.normal(100, 10, size=50)
    y_pred = y_true + rng.normal(0, 5, size=50)
    out = tmp_path / "residuals.png"

    plot_residuals(y_true, y_pred, str(out))

    assert out.exists() and out.stat().st_size > 0
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"
