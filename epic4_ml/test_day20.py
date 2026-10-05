"""Pytest suite for day20_trees_and_project.py and train.py.

Run with: pytest -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

file_dir = Path(__file__).parent
project_root = file_dir.parent
for p in (file_dir, project_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from epic4_ml.day20_trees_and_project import (
        depth_vs_score_analysis,
        load_experiment_data,
        plot_depth_vs_score,
        plot_feature_importance,
        plot_tree_diagram,
        save_model_bundle,
        train_and_compare_models,
    )
except ImportError:
    from day20_trees_and_project import (
        depth_vs_score_analysis,
        load_experiment_data,
        plot_depth_vs_score,
        plot_feature_importance,
        plot_tree_diagram,
        save_model_bundle,
        train_and_compare_models,
    )

TRAIN_PY = file_dir / "train.py"


def _assert_valid_png(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 0
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def classification_data():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    split = int(n * 0.8)
    return X[:split], X[split:], pd.Series(y[:split]), pd.Series(y[split:])


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 5))
    y = X[:, 0] * 3 + X[:, 1] * -2 + rng.normal(scale=0.5, size=n)
    split = int(n * 0.8)
    return X[:split], X[split:], pd.Series(y[:split]), pd.Series(y[split:])


# --------------------------------------------------------------------------
# depth_vs_score_analysis
# --------------------------------------------------------------------------
def test_depth_vs_score_returns_one_row_per_depth(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    depths = [2, 4, 6, None]
    result = depth_vs_score_analysis(X_train, X_test, y_train, y_test, depths, task="classification")
    assert len(result) == len(depths)
    assert list(result["max_depth"]) == [str(d) for d in depths]


def test_depth_vs_score_shows_overfitting_pattern(classification_data):
    # A fully-grown tree (max_depth=None) should fit training data almost
    # perfectly - that is the whole point of the overfitting demonstration.
    X_train, X_test, y_train, y_test = classification_data
    result = depth_vs_score_analysis(X_train, X_test, y_train, y_test, [2, None], task="classification")
    unlimited_row = result[result["max_depth"] == "None"].iloc[0]
    assert unlimited_row["train_score"] > 0.95


def test_depth_vs_score_works_for_regression(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    result = depth_vs_score_analysis(X_train, X_test, y_train, y_test, [2, 4], task="regression")
    assert len(result) == 2
    assert {"max_depth", "train_score", "test_score", "overfit_gap"} <= set(result.columns)


def test_depth_vs_score_rejects_empty_depths(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    with pytest.raises(ValueError):
        depth_vs_score_analysis(X_train, X_test, y_train, y_test, [], task="classification")


def test_depth_vs_score_rejects_invalid_task(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    with pytest.raises(ValueError):
        depth_vs_score_analysis(X_train, X_test, y_train, y_test, [2], task="clustering")


# --------------------------------------------------------------------------
# plot_depth_vs_score / plot_tree_diagram
# --------------------------------------------------------------------------
def test_plot_depth_vs_score_creates_valid_png(tmp_path, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    df = depth_vs_score_analysis(X_train, X_test, y_train, y_test, [2, 4], task="classification")
    out = tmp_path / "depth.png"
    plot_depth_vs_score(df, str(out))
    _assert_valid_png(out)


def test_plot_tree_diagram_creates_valid_png(tmp_path, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
    out = tmp_path / "tree.png"
    plot_tree_diagram(tree, [f"f{i}" for i in range(X_train.shape[1])], str(out), class_names=["0", "1"])
    _assert_valid_png(out)


# --------------------------------------------------------------------------
# plot_feature_importance
# --------------------------------------------------------------------------
def test_feature_importances_sum_to_approximately_one(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    forest = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
    assert forest.feature_importances_.sum() == pytest.approx(1.0, abs=1e-6)


def test_plot_feature_importance_creates_valid_png(tmp_path, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    forest = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
    out = tmp_path / "importance.png"
    plot_feature_importance(forest, [f"f{i}" for i in range(X_train.shape[1])], str(out))
    _assert_valid_png(out)


def test_plot_feature_importance_raises_for_non_tree_model():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit([[1], [2]], [0, 1])
    with pytest.raises(AttributeError):
        plot_feature_importance(model, ["f0"], "/tmp/wont_be_created.png")


# --------------------------------------------------------------------------
# train_and_compare_models / load_experiment_data (real datasets)
# --------------------------------------------------------------------------
def test_load_experiment_data_classification_shapes():
    data = load_experiment_data("classification")
    assert data["X_train"].shape[0] == len(data["y_train"])
    assert data["X_test"].shape[1] == data["X_train"].shape[1]
    assert len(data["feature_names"]) == data["X_train"].shape[1]


def test_train_and_compare_models_classification_has_expected_models():
    result = train_and_compare_models("classification")
    expected = {"dummy", "logistic_regression", "logistic_regression_balanced", "decision_tree", "random_forest"}
    assert expected <= set(result["model"])
    for col in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert result[col].between(0, 1).all()


def test_train_and_compare_models_regression_has_expected_models():
    result = train_and_compare_models("regression")
    expected = {"dummy", "linear_regression", "decision_tree", "random_forest"}
    assert expected <= set(result["model"])
    assert (result["rmse"] >= 0).all()


# --------------------------------------------------------------------------
# save_model_bundle
# --------------------------------------------------------------------------
def test_save_model_bundle_creates_both_files_and_reloads(tmp_path, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    metrics = {"accuracy": 0.9, "confusion_matrix": np.array([[1, 2], [3, 4]])}

    model_path, metrics_path = save_model_bundle(model, metrics, str(tmp_path))
    assert model_path.exists() and metrics_path.exists()

    with open(metrics_path) as f:
        loaded_metrics = json.load(f)  # must not raise - proves numpy types were sanitized
    assert loaded_metrics["confusion_matrix"] == [[1, 2], [3, 4]]

    import joblib
    reloaded_model = joblib.load(model_path)
    original_pred = model.predict(X_test)
    reloaded_pred = reloaded_model.predict(X_test)
    assert reloaded_pred.shape == original_pred.shape
    assert np.array_equal(reloaded_pred, original_pred)


# --------------------------------------------------------------------------
# train.py CLI (subprocess, exactly as a user would run it)
# --------------------------------------------------------------------------
def test_cli_runs_and_creates_expected_files(tmp_path):
    output_dir = tmp_path / "models"
    result = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--task", "classification", "--model", "decision_tree",
         "--output-dir", str(output_dir)],
        cwd=str(project_root), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()

    with open(output_dir / "metrics.json") as f:
        metrics = json.load(f)
    assert metrics["task"] == "classification"
    assert metrics["model"] == "decision_tree"
    assert 0 <= metrics["accuracy"] <= 1


def test_train_cli_main_runs_in_process(tmp_path):
    # Calling main() directly (vs. the subprocess test above) additionally
    # exercises train.py's own code paths for coverage purposes, while the
    # subprocess test above remains the proof that it genuinely works as a
    # real command-line tool end to end.
    try:
        from epic4_ml.train import main as train_main
    except ImportError:
        from train import main as train_main

    output_dir = tmp_path / "models_in_process"
    exit_code = train_main([
        "--task", "regression", "--model", "linear_regression", "--output-dir", str(output_dir),
    ])
    assert exit_code == 0
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()


def test_cli_rejects_invalid_model_choice(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--task", "classification", "--model", "not_a_real_model",
         "--output-dir", str(tmp_path / "out")],
        cwd=str(project_root), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert not (tmp_path / "out" / "model.joblib").exists()


# --------------------------------------------------------------------------
# End-to-end smoke tests: each day's own main() runs without exceptions.
# These exercise the full demonstration flow (real datasets, real charts)
# that a reader runs via `python epic4_mlX/dayNN_*.py` directly - not just
# the individual functions in isolation.
# --------------------------------------------------------------------------
def test_day16_main_runs_without_exceptions():
    try:
        from epic4_ml.day16_ml_basics import main as day16_main
    except ImportError:
        from day16_ml_basics import main as day16_main
    day16_main()  # must not raise


def test_day17_main_runs_without_exceptions():
    try:
        from epic4_ml.day17_feature_engineering import main as day17_main
    except ImportError:
        from day17_feature_engineering import main as day17_main
    day17_main()  # must not raise


def test_day18_main_runs_without_exceptions():
    try:
        from epic4_ml.day18_linear_regression import main as day18_main
    except ImportError:
        from day18_linear_regression import main as day18_main
    day18_main()  # must not raise


def test_day19_main_runs_without_exceptions():
    try:
        from epic4_ml.day19_logistic_regression import main as day19_main
    except ImportError:
        from day19_logistic_regression import main as day19_main
    day19_main()  # must not raise


def test_day20_main_runs_without_exceptions():
    try:
        from epic4_ml.day20_trees_and_project import main as day20_main
    except ImportError:
        from day20_trees_and_project import main as day20_main
    day20_main()  # must not raise - the full Epic 4 capstone demonstration
