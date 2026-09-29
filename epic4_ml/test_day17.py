"""Pytest suite for day17_feature_engineering.py.

Run with: pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

file_dir = Path(__file__).parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic4_ml.day17_feature_engineering import (
        ORDINAL_COLS,
        add_engineered_features,
        build_preprocessor,
        get_feature_names,
        load_preprocessor,
        load_raw_telco,
        save_preprocessor,
        select_top_features,
    )
except ImportError:
    from day17_feature_engineering import (
        ORDINAL_COLS,
        add_engineered_features,
        build_preprocessor,
        get_feature_names,
        load_preprocessor,
        load_raw_telco,
        save_preprocessor,
        select_top_features,
    )


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A small hand-built raw frame with real missing values in both a
    numeric column (TotalCharges) and a categorical column (PaymentMethod),
    shaped like load_raw_telco()'s output."""
    return pd.DataFrame({
        "gender": ["Female", "Male", "Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 0, 1, 0, 1, 0],
        "Partner": ["Yes", "No", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "No", "Yes"],
        "tenure": [1, 34, 0, 12, 60, 5],
        "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes", "No"],
        "MultipleLines": ["No phone service", "No", "No", "Yes", "Yes", "No phone service"],
        "InternetService": ["DSL", "DSL", "Fiber optic", "DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No", "Yes", "No", "No"],
        "OnlineBackup": ["Yes", "No", "No", "Yes", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No", "No", "Yes", "No"],
        "TechSupport": ["No", "No", "No", "Yes", "No", "No"],
        "StreamingTV": ["No", "No", "Yes", "No", "Yes", "No"],
        "StreamingMovies": ["No", "No", "Yes", "No", "Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "Two year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "Yes"],
        "PaymentMethod": ["Electronic check", np.nan, "Mailed check", "Bank transfer", np.nan, "Electronic check"],
        "MonthlyCharges": [29.85, 56.95, 52.55, 70.0, 100.0, 45.0],
        "TotalCharges": [29.85, 1889.5, np.nan, 840.0, 6000.0, 225.0],
    })


@pytest.fixture
def numeric_cols():
    return ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
            "tenure_years", "avg_monthly_spend", "monthly_charge_x_tenure"]


@pytest.fixture
def nominal_cols():
    return ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "PaymentMethod"]


# --------------------------------------------------------------------------
# add_engineered_features
# --------------------------------------------------------------------------
def test_add_engineered_features_does_not_mutate_input(raw_frame):
    original_columns = list(raw_frame.columns)
    original_copy = raw_frame.copy()
    add_engineered_features(raw_frame)
    assert list(raw_frame.columns) == original_columns
    pd.testing.assert_frame_equal(raw_frame, original_copy)


def test_add_engineered_features_adds_at_least_four_columns(raw_frame):
    result = add_engineered_features(raw_frame)
    new_cols = set(result.columns) - set(raw_frame.columns)
    assert len(new_cols) >= 4
    assert {"tenure_years", "avg_monthly_spend", "tenure_bucket", "monthly_charge_x_tenure"} <= new_cols


def test_add_engineered_features_values_are_correct(raw_frame):
    result = add_engineered_features(raw_frame)
    assert result["tenure_years"].iloc[1] == pytest.approx(34 / 12.0)
    assert result["monthly_charge_x_tenure"].iloc[1] == pytest.approx(56.95 * 34)


def test_add_engineered_features_avoids_divide_by_zero_at_tenure_zero():
    # A brand-new customer: tenure=0 with a real (non-NaN) TotalCharges.
    # Naively dividing by tenure would produce inf; the function must guard
    # against that rather than silently returning inf or crashing.
    df = pd.DataFrame({"tenure": [0], "MonthlyCharges": [50.0], "TotalCharges": [0.0]})
    result = add_engineered_features(df)
    assert np.isfinite(result["avg_monthly_spend"].iloc[0])


def test_add_engineered_features_is_leakage_safe_train_vs_full(raw_frame):
    # Calling it on a subset must give the SAME values for those rows as
    # calling it on the full frame - proof no statistic is being recomputed
    # per-subset (which would silently differ between train/test calls).
    subset = raw_frame.iloc[[1, 3, 4]]
    full_result = add_engineered_features(raw_frame).iloc[[1, 3, 4]].reset_index(drop=True)
    subset_result = add_engineered_features(subset).reset_index(drop=True)
    pd.testing.assert_frame_equal(full_result, subset_result)


# --------------------------------------------------------------------------
# build_preprocessor / get_feature_names
# --------------------------------------------------------------------------
def test_preprocessor_output_has_no_nans(raw_frame, numeric_cols, nominal_cols):
    fe = add_engineered_features(raw_frame)
    preprocessor = build_preprocessor(numeric_cols, nominal_cols)
    transformed = preprocessor.fit_transform(fe)
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unseen_category_without_crashing(raw_frame, numeric_cols, nominal_cols):
    fe = add_engineered_features(raw_frame)
    preprocessor = build_preprocessor(numeric_cols, nominal_cols).fit(fe)

    unseen = fe.iloc[[0]].copy()
    unseen["PaymentMethod"] = "Cryptocurrency"  # never seen during fit
    unseen["InternetService"] = "Satellite"      # never seen during fit
    transformed = preprocessor.transform(unseen)  # must not raise
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == 1


def test_preprocessor_ordinal_encoder_preserves_contract_order(raw_frame, numeric_cols, nominal_cols):
    fe = add_engineered_features(raw_frame)
    preprocessor = build_preprocessor(numeric_cols, nominal_cols).fit(fe)
    names = get_feature_names(preprocessor)
    contract_col_idx = names.index("Contract")

    two_year_row = fe[fe["Contract"] == "Two year"].iloc[[0]]
    month_to_month_row = fe[fe["Contract"] == "Month-to-month"].iloc[[0]]
    two_year_encoded = preprocessor.transform(two_year_row)[0, contract_col_idx]
    mtm_encoded = preprocessor.transform(month_to_month_row)[0, contract_col_idx]
    assert two_year_encoded > mtm_encoded  # "Two year" ranks above "Month-to-month"


def test_get_feature_names_strips_transformer_prefix_and_matches_shape(raw_frame, numeric_cols, nominal_cols):
    fe = add_engineered_features(raw_frame)
    preprocessor = build_preprocessor(numeric_cols, nominal_cols).fit(fe)
    names = get_feature_names(preprocessor)
    transformed = preprocessor.transform(fe)
    assert len(names) == transformed.shape[1]
    assert not any("__" in name for name in names)


def test_build_preprocessor_invalid_scaler_raises():
    with pytest.raises(ValueError):
        build_preprocessor(["a"], ["b"], scaler="robust")


def test_build_preprocessor_minmax_scales_into_zero_one_range(raw_frame, numeric_cols, nominal_cols):
    fe = add_engineered_features(raw_frame)
    preprocessor = build_preprocessor(numeric_cols, nominal_cols, scaler="minmax").fit(fe)
    names = get_feature_names(preprocessor)
    transformed = preprocessor.transform(fe)
    tenure_col = transformed[:, names.index("tenure")]
    assert tenure_col.min() >= 0.0 and tenure_col.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# select_top_features
# --------------------------------------------------------------------------
def test_select_top_features_returns_k_names():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "informative": rng.normal(size=200),
        "noise_1": rng.normal(size=200),
        "noise_2": rng.normal(size=200),
    })
    y = (X["informative"] + rng.normal(scale=0.1, size=200) > 0).astype(int)
    top = select_top_features(X, y, k=1, task="classification")
    assert top == ["informative"]


def test_select_top_features_caps_k_at_column_count():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1])
    result = select_top_features(X, y, k=10, task="classification")
    assert len(result) == 2


def test_select_top_features_invalid_k_raises():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    with pytest.raises(ValueError):
        select_top_features(X, y, k=0)


# --------------------------------------------------------------------------
# save_preprocessor / load_preprocessor
# --------------------------------------------------------------------------
def test_saved_and_reloaded_preprocessor_gives_identical_output(tmp_path, raw_frame, numeric_cols, nominal_cols):
    fe = add_engineered_features(raw_frame)
    preprocessor = build_preprocessor(numeric_cols, nominal_cols).fit(fe)
    original_output = preprocessor.transform(fe)

    save_path = tmp_path / "preprocessor.joblib"
    save_preprocessor(preprocessor, save_path)
    assert save_path.exists()

    reloaded = load_preprocessor(save_path)
    reloaded_output = reloaded.transform(fe)
    assert np.allclose(original_output, reloaded_output)


# --------------------------------------------------------------------------
# load_raw_telco
# --------------------------------------------------------------------------
def test_load_raw_telco_has_real_missing_values(tmp_path, monkeypatch):
    fake_raw = pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "TotalCharges": ["10.0", " ", "30.0"],
        "Churn": ["No", "Yes", "No"],
        "gender": ["Female", "Male", "Female"],
    })
    module = sys.modules[load_raw_telco.__module__]
    monkeypatch.setattr(module, "_download_telco_raw", lambda: fake_raw)

    df = load_raw_telco(data_dir=tmp_path, inject_missing=False)
    assert "customerID" not in df.columns
    assert df["Churn"].tolist() == [0, 1, 0]
    assert df["TotalCharges"].isna().sum() == 1  # the blank became a real NaN, not 0.0


def test_load_raw_telco_caches_to_csv(tmp_path, monkeypatch):
    fake_raw = pd.DataFrame({
        "customerID": ["A", "B"], "TotalCharges": ["10.0", "20.0"],
        "Churn": ["No", "No"], "gender": ["Female", "Male"],
    })
    calls = {"n": 0}

    def fake_download():
        calls["n"] += 1
        return fake_raw

    module = sys.modules[load_raw_telco.__module__]
    monkeypatch.setattr(module, "_download_telco_raw", fake_download)

    load_raw_telco(data_dir=tmp_path, inject_missing=False)
    load_raw_telco(data_dir=tmp_path, inject_missing=False)
    assert calls["n"] == 1
    assert (tmp_path / "telco_churn_raw.csv").exists()
