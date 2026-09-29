"""Day 17: Feature Engineering & Preprocessing Pipelines - Epic 4.

Dependency: reuses Day 16's `split_data` (80/20, stratified, random_state=42)
and extends the same `MLExperiment` object with a fitted preprocessor,
rather than building a separate one-off pipeline disconnected from
yesterday's work.

Today works on the RAW Telco Customer Churn columns (strings, real missing
values) rather than Day 16's already-cleaned, already-one-hot-encoded `X` -
there is nothing left to engineer or impute once a column is already numeric,
so Day 17 needs the dataset one step further upstream. `load_raw_telco()`
below is that upstream version.

=============================================================================
StandardScaler vs. MinMaxScaler - when to prefer which
=============================================================================
StandardScaler: (x - mean) / std. Centers data at 0 with unit variance.
    Preferred for: linear/logistic regression, SVMs, PCA, anything that
    assumes roughly Gaussian-shaped features, and as the general default.
    It is also less thrown off by a single extreme outlier than MinMax,
    because one huge value only stretches the standard deviation a little,
    whereas it stretches MinMax's whole [0, 1] range around that one point.

MinMaxScaler: (x - min) / (max - min). Squashes data into a fixed [0, 1]
    range. Preferred for: neural networks (many activation functions
    expect bounded input), image-pixel-style data, or any algorithm that
    needs a strictly bounded range rather than an assumption of Gaussian
    shape. Downside: it is very sensitive to outliers, since a single
    extreme min or max rescales every other point.

Rule of thumb used below: StandardScaler is the default for this project's
numeric columns (tenure, charges) since they feed linear/KNN-style models
later, not a neural net.
=============================================================================
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from epic4_ml.day16_ml_basics import DATA_DIR, MLExperiment, _download_telco_raw, split_data
except ImportError:
    from day16_ml_basics import DATA_DIR, MLExperiment, _download_telco_raw, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# Contract has a genuine natural order (longer commitment = "more"), unlike
# gender/PaymentMethod/etc, which have no inherent ranking - that's the
# distinction OrdinalEncoder vs. OneHotEncoder is meant to capture.
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
TENURE_BUCKET_ORDER = ["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
                "tenure_years", "avg_monthly_spend", "monthly_charge_x_tenure"]
ORDINAL_COLS = {"Contract": CONTRACT_ORDER, "tenure_bucket": TENURE_BUCKET_ORDER}
NOMINAL_COLS = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "PaymentMethod"]


# --------------------------------------------------------------------------
# 0. Raw dataset loader (upstream of Day 16's already-encoded version)
# --------------------------------------------------------------------------
def load_raw_telco(data_dir: Optional[Path] = None, inject_missing: bool = True) -> pd.DataFrame:
    """Load the Telco Customer Churn data in its RAW, unencoded form.

    Unlike Day 16's `load_ml_dataset("telco_churn")` (which already fills
    missing values and one-hot encodes everything, so there is nothing
    left for today's SimpleImputer/OneHotEncoder to do), this returns:
        - `TotalCharges` as a float column WITH real NaNs (11 customers in
          the real data have a blank TotalCharges - all have tenure 0,
          i.e. brand-new customers never billed yet).
        - Every categorical column still as text (e.g. "Month-to-month"),
          not one-hot encoded.
        - `Churn` mapped to 0/1 and kept as a separate return is NOT done
          here - callers split X/y themselves (see `main()`), matching how
          a real raw extract arrives.

    `inject_missing=True` (default) additionally blanks out ~2% of the
    `PaymentMethod` column with a fixed seed. The real Telco CSV has zero
    missing categorical values, so without this there would be nothing to
    demonstrate `SimpleImputer(strategy="most_frequent")` ON - this mirrors
    Day 8's deliberately-added "Garden & Outdoor" category: a small,
    clearly-documented synthetic tweak that makes an otherwise
    untestable code path exercisable.

    Args:
        data_dir: Folder for the raw CSV cache (default: epic4_ml/data/).
        inject_missing: Whether to synthetically blank ~2% of PaymentMethod.

    Returns:
        DataFrame with `customerID` dropped and `Churn` mapped to 0/1,
        every other raw column intact.
    """
    folder = Path(data_dir) if data_dir is not None else DATA_DIR
    raw_path = folder / "telco_churn_raw.csv"

    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        df = _download_telco_raw()
        df = df.drop(columns=["customerID"], errors="ignore")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> real NaN
        df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)

        if inject_missing:
            rng = np.random.default_rng(42)
            missing_idx = rng.choice(df.index, size=max(1, int(0.02 * len(df))), replace=False)
            df.loc[missing_idx, "PaymentMethod"] = np.nan

        folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        logger.info("Saved raw Telco dataset to %s", raw_path)

    return df


# --------------------------------------------------------------------------
# 1. Feature engineering
# --------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 4 new features derived from the raw Telco columns. Does NOT mutate `df`.

    New columns (each addresses something the raw columns don't capture
    directly):
        1. tenure_years       - tenure (months) / 12. A ratio/unit-conversion
                                 feature: puts tenure on a smaller, more
                                 interpretable scale for models sensitive
                                 to feature magnitude.
        2. avg_monthly_spend  - TotalCharges / max(tenure, 1). The
                                 customer's historical average monthly bill,
                                 which can differ from their CURRENT
                                 MonthlyCharges if their plan changed over
                                 time - the gap between the two is itself
                                 informative and invisible in either raw
                                 column alone.
        3. tenure_bucket      - tenure binned into lifecycle stages
                                 ("0-1yr" ... "5yr+"). Captures non-linear
                                 lifecycle effects (e.g. a cliff in churn
                                 risk right after the first year) that a
                                 single linear tenure coefficient can't.
        4. monthly_charge_x_tenure - MonthlyCharges * tenure. An interaction
                                 term approximating revenue-to-date, distinct
                                 from either factor alone (a long-tenure,
                                 low-charge customer and a short-tenure,
                                 high-charge customer can have the same
                                 tenure or the same charge, but very
                                 different values here).

    All four use only fixed arithmetic/bucket boundaries - no statistic
    (mean, median, quantile) is computed FROM `df` itself, so this function
    gives identical, leakage-safe results whether it's called on the full
    dataset, the train split, or the test split separately.

    Args:
        df: Input DataFrame with at least 'tenure', 'MonthlyCharges', and
            'TotalCharges' columns.

    Returns:
        A NEW DataFrame (`df` is left unmodified) with the 4 columns added.
    """
    result = df.copy()

    safe_tenure = result["tenure"].replace(0, 1)  # avoid dividing by zero for brand-new customers
    result["tenure_years"] = result["tenure"] / 12.0
    result["avg_monthly_spend"] = result["TotalCharges"] / safe_tenure
    result["tenure_bucket"] = pd.cut(
        result["tenure"], bins=[-1, 12, 24, 48, 60, np.inf], labels=TENURE_BUCKET_ORDER,
    ).astype(str)
    result["monthly_charge_x_tenure"] = result["MonthlyCharges"] * result["tenure"]

    return result


# --------------------------------------------------------------------------
# 2. Preprocessing pipeline
# --------------------------------------------------------------------------
def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    ordinal_cols: Optional[Dict[str, List[str]]] = None,
    scaler: str = "standard",
) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes, scales, and encodes in one fitted object.

    - Numeric columns: median imputation (robust to outliers, unlike mean)
      then scaling (`scaler="standard"` -> StandardScaler, `"minmax"` ->
      MinMaxScaler; see the module docstring for when to prefer which).
    - Nominal categorical columns (`categorical_cols`, no inherent order):
      most-frequent imputation, then OneHotEncoder with
      `handle_unknown="ignore"` so a category never seen during fit (e.g.
      a brand-new PaymentMethod at inference time) produces all-zero
      columns instead of crashing.
    - Ordinal categorical columns (`ordinal_cols`, a dict of column name ->
      ordered category list, e.g. Contract's Month-to-month < One year <
      Two year): most-frequent imputation, then OrdinalEncoder with that
      exact category order, plus `handle_unknown="use_encoded_value"` so
      an unseen category maps to -1 instead of crashing.

    Args:
        numeric_cols: Numeric column names.
        categorical_cols: Nominal (unordered) categorical column names.
        ordinal_cols: Dict of ordinal column name -> its category order,
            smallest/first to largest/last. Defaults to this project's
            Contract + tenure_bucket columns.
        scaler: "standard" (default) or "minmax".

    Returns:
        An unfitted `ColumnTransformer` (call `.fit`/`.fit_transform` on it).

    Raises:
        ValueError: If `scaler` is not "standard" or "minmax".
    """
    if scaler == "standard":
        scaler_instance = StandardScaler()
    elif scaler == "minmax":
        scaler_instance = MinMaxScaler()
    else:
        raise ValueError(f"scaler must be 'standard' or 'minmax', got {scaler!r}")

    if ordinal_cols is None:
        ordinal_cols = ORDINAL_COLS

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_instance),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        nominal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", nominal_pipeline, categorical_cols))

    if ordinal_cols:
        ordinal_col_names = [c for c in ordinal_cols if c in (categorical_cols or []) or c not in (numeric_cols or [])]
        ordinal_col_names = list(ordinal_cols.keys())
        category_order = [ordinal_cols[col] for col in ordinal_col_names]
        ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=category_order, handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("ord", ordinal_pipeline, ordinal_col_names))

    return ColumnTransformer(transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Map a FITTED preprocessor's transformed columns back to readable names.

    `ColumnTransformer.get_feature_names_out()` returns names prefixed with
    the transformer's label (e.g. "num__tenure", "cat__gender_Male") - this
    strips that prefix so the returned names read like plain column names
    (e.g. "tenure", "gender_Male"), while still being unique and in the
    exact column order of the transformed matrix.

    Args:
        preprocessor: A FITTED ColumnTransformer (e.g. from
            `build_preprocessor(...).fit(X_train)`).

    Returns:
        List of readable feature names, one per output column.
    """
    raw_names = preprocessor.get_feature_names_out()
    return [name.split("__", 1)[1] if "__" in name else name for name in raw_names]


# --------------------------------------------------------------------------
# 3. Feature selection
# --------------------------------------------------------------------------
def select_top_features(X: pd.DataFrame, y: pd.Series, k: int, task: str = "auto") -> List[str]:
    """Pick the top-`k` columns of an already-NUMERIC feature matrix by univariate score.

    Uses `f_classif` (ANOVA F-value) for classification targets or
    `f_regression` for regression targets - both are fast univariate
    filters: "how much does this ONE column, alone, relate to the target?"
    They don't capture interactions between features the way a full model
    would, which is exactly why this is a quick pre-filter, not a
    replacement for proper model-based feature importance.

    Args:
        X: Numeric feature matrix (DataFrame with column names, or ndarray -
            if an ndarray, generic names "feature_0", "feature_1"... are
            used).
        y: Target vector.
        k: Number of top features to keep (capped at the number of
            available columns).
        task: "classification", "regression", or "auto" (default) to guess
            from `y`'s dtype/cardinality, the same rule Day 16 uses.

    Returns:
        List of the top-`k` column names, ordered by descending score.

    Raises:
        ValueError: If `k` is not a positive integer.
    """
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    if isinstance(X, pd.DataFrame):
        columns = list(X.columns)
        X_values = X.to_numpy(dtype=float)
    else:
        X_values = np.asarray(X, dtype=float)
        columns = [f"feature_{i}" for i in range(X_values.shape[1])]

    if task == "auto":
        is_integer_like = pd.api.types.is_integer_dtype(y) or (pd.Series(y).nunique() <= 20)
        task = "classification" if is_integer_like else "regression"

    score_func = f_classif if task == "classification" else f_regression
    k = min(k, X_values.shape[1])

    selector = SelectKBest(score_func=score_func, k=k).fit(X_values, y)
    selected_mask = selector.get_support()
    scores = selector.scores_

    ranked = sorted((columns[i] for i in range(len(columns)) if selected_mask[i]),
                     key=lambda name: scores[columns.index(name)], reverse=True)
    return ranked


# --------------------------------------------------------------------------
# 4. Persistence
# --------------------------------------------------------------------------
def save_preprocessor(preprocessor: ColumnTransformer, path: Optional[Path] = None) -> Path:
    """Save a FITTED preprocessor with joblib. Returns the path written to."""
    target = Path(path) if path is not None else PREPROCESSOR_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, target)
    logger.info("Saved fitted preprocessor: %s", target)
    return target


def load_preprocessor(path: Optional[Path] = None) -> ColumnTransformer:
    """Load a preprocessor previously saved with `save_preprocessor`."""
    source = Path(path) if path is not None else PREPROCESSOR_PATH
    return joblib.load(source)


# --------------------------------------------------------------------------
# 5. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 17: Feature Engineering & Preprocessing Pipelines ===")

    raw = load_raw_telco()
    y = raw["Churn"]
    X_raw = raw.drop(columns=["Churn"])
    logger.info("Raw data: %s, missing values per column (top 5):\n%s",
                X_raw.shape, X_raw.isna().sum().sort_values(ascending=False).head(5))

    # Reuse Day 16's exact split (same test_size/random_state/stratify rule),
    # and the same MLExperiment object this dataset already has.
    X_train_raw, X_test_raw, y_train, y_test = split_data(X_raw, y, test_size=0.2, stratify=True)

    X_train_fe = add_engineered_features(X_train_raw)
    X_test_fe = add_engineered_features(X_test_raw)
    logger.info("After feature engineering: %s -> %s columns", X_train_raw.shape[1], X_train_fe.shape[1])

    preprocessor = build_preprocessor(NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS)
    X_train_transformed = preprocessor.fit_transform(X_train_fe, y_train)
    X_test_transformed = preprocessor.transform(X_test_fe)

    feature_names = get_feature_names(preprocessor)
    logger.info(
        "Preprocessor fitted on train only. Shapes: train %s -> %s, test %s -> %s",
        X_train_fe.shape, X_train_transformed.shape, X_test_fe.shape, X_test_transformed.shape,
    )
    logger.info("NaNs remaining: train=%d, test=%d",
                np.isnan(X_train_transformed).sum(), np.isnan(X_test_transformed).sum())
    logger.info("First 10 readable feature names: %s", feature_names[:10])

    top_features = select_top_features(
        pd.DataFrame(X_train_transformed, columns=feature_names), y_train, k=10, task="classification",
    )
    logger.info("Top 10 features by SelectKBest (f_classif): %s", top_features)

    saved_path = save_preprocessor(preprocessor)
    reloaded = load_preprocessor(saved_path)
    reloaded_output = reloaded.transform(X_test_fe)
    identical = np.allclose(X_test_transformed, reloaded_output)
    logger.info("Reloaded preprocessor output identical to original: %s", identical)

    experiment = MLExperiment("telco_churn")
    experiment.X_train, experiment.X_test, experiment.y_train, experiment.y_test = (
        X_train_transformed, X_test_transformed, y_train, y_test,
    )
    experiment.set_preprocessor(preprocessor)
    logger.info("Attached fitted preprocessor to the telco_churn MLExperiment for reuse later this week.")


if __name__ == "__main__":
    main()
