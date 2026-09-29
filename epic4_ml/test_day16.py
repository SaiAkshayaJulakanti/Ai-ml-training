"""Pytest suite for day16_ml_basics.py.

Run with: pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

file_dir = Path(__file__).parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic4_ml.day16_ml_basics import (
        DATA_DIR,
        MLExperiment,
        _clean_telco_frame,
        class_distribution,
        demonstrate_leakage,
        get_baseline_scores,
        load_ml_dataset,
        split_data,
    )
except ImportError:
    from day16_ml_basics import (
        DATA_DIR,
        MLExperiment,
        _clean_telco_frame,
        class_distribution,
        demonstrate_leakage,
        get_baseline_scores,
        load_ml_dataset,
        split_data,
    )


@pytest.fixture(scope="module")
def cancer(tmp_path_factory):
    """Breast cancer (bundled with scikit-learn, no download needed)."""
    return load_ml_dataset("breast_cancer", data_dir=tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="module")
def diabetes(tmp_path_factory):
    return load_ml_dataset("diabetes", data_dir=tmp_path_factory.mktemp("data"))


@pytest.fixture
def imbalanced():
    """1000 rows, 90/10 class imbalance - a case where stratifying really matters."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000)})
    y = pd.Series([0] * 900 + [1] * 100, name="target")
    return X, y


# --------------------------------------------------------------------------
# load_ml_dataset
# --------------------------------------------------------------------------
def test_load_ml_dataset_returns_features_and_target(diabetes):
    X, y = diabetes
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)
    assert X.shape == (442, 10)
    assert len(y) == 442
    assert "target" not in X.columns


def test_load_ml_dataset_saves_csv_and_reloads_identically(tmp_path):
    X1, y1 = load_ml_dataset("breast_cancer", data_dir=tmp_path)
    assert (tmp_path / "breast_cancer.csv").exists()
    X2, y2 = load_ml_dataset("breast_cancer", data_dir=tmp_path)  # now read from the CSV
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_load_ml_dataset_unknown_name_raises():
    with pytest.raises(ValueError):
        load_ml_dataset("not_a_dataset")


def test_load_ml_dataset_california_housing_uses_cache_after_first_load(tmp_path, monkeypatch):
    # California Housing needs a download, so fake the fetch and verify
    # the caching logic: fetched once, saved as CSV, then read from disk.
    calls = {"n": 0}

    def fake_fetch(as_frame=True):
        calls["n"] += 1
        return SimpleNamespace(
            data=pd.DataFrame({"MedInc": [1.0, 2.0, 3.0], "HouseAge": [10.0, 20.0, 30.0]}),
            target=pd.Series([0.5, 1.5, 2.5]),
        )

    monkeypatch.setattr("sklearn.datasets.fetch_california_housing", fake_fetch)
    X, y = load_ml_dataset("california_housing", data_dir=tmp_path)
    load_ml_dataset("california_housing", data_dir=tmp_path)
    assert calls["n"] == 1
    assert list(X.columns) == ["MedInc", "HouseAge"]
    assert list(y) == [0.5, 1.5, 2.5]


# --------------------------------------------------------------------------
# split_data
# --------------------------------------------------------------------------
def test_split_sizes_are_80_20(diabetes):
    X, y = diabetes
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) == pytest.approx(0.2 * len(X), abs=1)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)


def test_stratified_class_ratios_match_within_2_percent(cancer):
    X, y = cancer
    _, _, y_train, y_test = split_data(X, y, test_size=0.2, stratify=True)
    full = class_distribution(y)
    for part in (class_distribution(y_train), class_distribution(y_test)):
        for cls in full.index:
            assert abs(part[cls] - full[cls]) < 0.02


def test_stratified_ratios_hold_on_imbalanced_data(imbalanced):
    X, y = imbalanced
    _, _, y_train, y_test = split_data(X, y, test_size=0.2, stratify=True)
    assert y_train.mean() == pytest.approx(0.10, abs=0.005)
    assert y_test.mean() == pytest.approx(0.10, abs=0.005)


def test_split_is_reproducible_with_fixed_random_state(diabetes):
    X, y = diabetes
    first = split_data(X, y, test_size=0.2)
    second = split_data(X, y, test_size=0.2)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_series_equal(first[3], second[3])


@pytest.mark.parametrize("bad_test_size", [0, 1, -0.1, 1.5, "0.2", None, True])
def test_invalid_test_size_raises_value_error(diabetes, bad_test_size):
    X, y = diabetes
    with pytest.raises(ValueError):
        split_data(X, y, test_size=bad_test_size)


def test_empty_data_raises_value_error():
    with pytest.raises(ValueError):
        split_data(pd.DataFrame({"a": []}), pd.Series([], dtype=float), test_size=0.2)


def test_mismatched_lengths_raise_value_error(diabetes):
    X, y = diabetes
    with pytest.raises(ValueError):
        split_data(X, y.iloc[:-5], test_size=0.2)


# --------------------------------------------------------------------------
# get_baseline_scores
# --------------------------------------------------------------------------
def test_regression_baseline_returns_expected_keys(diabetes):
    X, y = diabetes
    scores = get_baseline_scores(*split_data(X, y), task="regression")
    expected = {"task", "strategy", "primary_metric", "primary_score", "train_r2",
                "test_r2", "test_rmse", "test_mae", "baseline_prediction"}
    assert expected <= set(scores)
    assert scores["strategy"] == "mean"


def test_classification_baseline_returns_expected_keys(cancer):
    X, y = cancer
    scores = get_baseline_scores(*split_data(X, y, stratify=True), task="classification")
    expected = {"task", "strategy", "primary_metric", "primary_score", "train_accuracy",
                "test_accuracy", "test_f1", "majority_class"}
    assert expected <= set(scores)
    assert scores["strategy"] == "most_frequent"


def test_regression_baseline_r2_is_about_zero(diabetes):
    # Predicting the training mean can't explain any variance: R^2 ~ 0.
    X, y = diabetes
    scores = get_baseline_scores(*split_data(X, y), task="regression")
    assert scores["train_r2"] == pytest.approx(0.0, abs=1e-9)
    assert abs(scores["test_r2"]) < 0.1


def test_classification_baseline_accuracy_equals_majority_share(cancer):
    X, y = cancer
    X_train, X_test, y_train, y_test = split_data(X, y, stratify=True)
    scores = get_baseline_scores(X_train, X_test, y_train, y_test, task="classification")
    majority = y_train.value_counts().idxmax()
    assert scores["majority_class"] == majority
    assert scores["test_accuracy"] == pytest.approx((y_test == majority).mean())


def test_baseline_rejects_unknown_task(diabetes):
    X, y = diabetes
    with pytest.raises(ValueError):
        get_baseline_scores(*split_data(X, y), task="clustering")


# --------------------------------------------------------------------------
# demonstrate_leakage
# --------------------------------------------------------------------------
def test_leakage_returns_expected_keys_for_both_tasks(cancer, diabetes):
    expected = {"task", "model", "metric", "leaky_score", "clean_score", "score_difference", "scaler_mean_gap"}
    assert expected <= set(demonstrate_leakage(*cancer))
    assert expected <= set(demonstrate_leakage(*diabetes))


def test_leakage_detects_task_from_target(cancer, diabetes):
    assert demonstrate_leakage(*cancer)["task"] == "classification"
    assert demonstrate_leakage(*diabetes)["task"] == "regression"


def test_leakage_scaler_statistics_measurably_differ(cancer, diabetes):
    # The leaky scaler saw the test rows, so its mean must differ from the
    # train-only scaler's mean - that gap IS the leakage.
    for dataset in (cancer, diabetes):
        result = demonstrate_leakage(*dataset)
        assert result["scaler_mean_gap"] > 0
        assert result["score_difference"] == pytest.approx(result["leaky_score"] - result["clean_score"])


# --------------------------------------------------------------------------
# MLExperiment
# --------------------------------------------------------------------------
def test_ml_experiment_run_records_split_and_baseline(tmp_path):
    exp = MLExperiment("breast_cancer", data_dir=tmp_path).run()
    assert exp.n_train + exp.n_test == 569
    assert exp.n_features == 30
    assert exp.stratified is True
    assert exp.baseline_score == pytest.approx(exp.baseline_scores["test_accuracy"])


def test_ml_experiment_summary_contents_and_recorded_models(tmp_path):
    exp = MLExperiment("diabetes", data_dir=tmp_path)
    assert "not run yet" in exp.summary()

    exp.run()
    exp.record_model("LinearRegression", 0.45)
    summary = exp.summary()
    assert "diabetes" in summary and "regression" in summary
    assert f"{exp.n_train} train / {exp.n_test} test" in summary
    assert "LinearRegression" in summary and "vs baseline" in summary


def test_ml_experiment_unknown_dataset_raises():
    with pytest.raises(ValueError):
        MLExperiment("not_a_dataset")


# --------------------------------------------------------------------------
# Telco Customer Churn (the classification dataset used in main())
# --------------------------------------------------------------------------
@pytest.fixture
def raw_telco():
    """A tiny hand-made table shaped like the raw Telco CSV."""
    return pd.DataFrame({
        "customerID": ["A-1", "B-2", "C-3", "D-4"],
        "gender": ["Female", "Male", "Male", "Female"],
        "tenure": [1, 34, 0, 12],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "MonthlyCharges": [29.85, 56.95, 52.55, 70.0],
        "TotalCharges": ["29.85", "1889.5", " ", "840.0"],   # blank = never billed
        "Churn": ["No", "No", "No", "Yes"],
    })


def test_clean_telco_frame_encodes_everything_numeric(raw_telco):
    X, y = _clean_telco_frame(raw_telco)
    assert "customerID" not in X.columns and "Churn" not in X.columns
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)
    assert list(y) == [0, 0, 0, 1]
    assert len(X) == len(y) == 4


def test_clean_telco_frame_fixes_blank_total_charges(raw_telco):
    X, _ = _clean_telco_frame(raw_telco)
    assert X["TotalCharges"].tolist() == [29.85, 1889.5, 0.0, 840.0]
    assert not X.isna().any().any()


def test_load_telco_downloads_once_then_uses_cached_csv(tmp_path, monkeypatch, raw_telco):
    module = sys.modules[load_ml_dataset.__module__]
    calls = {"n": 0}

    def fake_download():
        calls["n"] += 1
        return raw_telco

    monkeypatch.setattr(module, "_download_telco_raw", fake_download)
    X1, y1 = load_ml_dataset("telco_churn", data_dir=tmp_path)
    X2, y2 = load_ml_dataset("telco_churn", data_dir=tmp_path)
    assert calls["n"] == 1
    assert (tmp_path / "telco_churn.csv").exists()
    pd.testing.assert_frame_equal(X1, X2)
    assert list(y2) == [0, 0, 0, 1]


telco_csv_present = (DATA_DIR / "telco_churn.csv").exists()


@pytest.mark.skipif(not telco_csv_present, reason="epic4_ml/data/telco_churn.csv not present")
def test_real_telco_dataset_shape_and_churn_rate():
    X, y = load_ml_dataset("telco_churn")
    assert len(X) == len(y) == 7043
    assert set(y.unique()) == {0, 1}
    assert y.mean() == pytest.approx(0.2654, abs=0.001)


@pytest.mark.skipif(not telco_csv_present, reason="epic4_ml/data/telco_churn.csv not present")
def test_real_telco_stratified_split_keeps_churn_rate_within_2_percent():
    X, y = load_ml_dataset("telco_churn")
    _, _, y_train, y_test = split_data(X, y, test_size=0.2, stratify=True)
    assert y_train.mean() == pytest.approx(y.mean(), abs=0.02)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.02)
    assert len(y_test) == pytest.approx(0.2 * len(y), abs=1)


@pytest.mark.skipif(not telco_csv_present, reason="epic4_ml/data/telco_churn.csv not present")
def test_real_telco_baseline_high_accuracy_but_zero_f1():
    # The classic imbalanced-data trap: always predicting "no churn" scores
    # ~73% accuracy while catching zero churners.
    X, y = load_ml_dataset("telco_churn")
    scores = get_baseline_scores(*split_data(X, y, stratify=True), task="classification")
    assert scores["majority_class"] == 0
    assert scores["test_accuracy"] == pytest.approx(0.7346, abs=0.005)
    assert scores["test_f1"] == 0.0
