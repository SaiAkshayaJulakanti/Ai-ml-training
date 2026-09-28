"""Day 16: ML Fundamentals, Train/Test Split & Baseline - Epic 4.

Dependency: Epic 3 is complete (`epic3-complete` tag) and its habits carry
over: validate inputs, test everything, and never trust a number you
haven't compared against something.

The ML workflow this file walks through, in order:
    data -> features/target -> train/test split -> baseline -> (real models, later)

Datasets used (both saved as CSV under `epic4_ml/data/` on first load):
    - Regression:     "diabetes"       (442 rows, predict disease progression)
    - Classification: "breast_cancer"  (569 rows, malignant vs. benign)
    - Also supported: "california_housing" (20,640 rows, regression). It is
      downloaded by scikit-learn on first use, so it needs internet access
      the first time; after that it is cached as a CSV like the others.

=============================================================================
Overfitting, underfitting, and bias vs. variance (in plain words)
=============================================================================
(Draft explanation - reword it in your own words before you submit it.)

Underfitting: the model is too simple to learn the real pattern, so it does
    badly on the training data AND on new data. Like studying only the
    chapter titles before an exam. (High bias.)

Overfitting: the model is so flexible that it memorises the training data,
    including its random noise, so it looks great on training data but
    falls apart on new data. Like memorising last year's exam answers
    word for word instead of learning the subject. (High variance.)

Bias vs. variance: bias is error from being too simple / making wrong
    assumptions; variance is error from being too sensitive to whichever
    particular rows happened to be in the training set. Making a model
    more complex usually lowers bias but raises variance, so the goal is
    the sweet spot where error on NEW data is smallest. That is exactly why
    we hold out a test set: training score alone cannot tell us which side
    of the sweet spot we are on.

Why baselines matter: a Dummy model that ignores the features entirely sets
    the score any real model has to beat. If a fancy model barely beats the
    baseline, it has not really learned much. (Note the classification
    trap: on breast cancer, always guessing the majority class already
    scores ~63% accuracy, so "63% accuracy" is NOT impressive there.)
=============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

REGRESSION_DATASET = "diabetes"
CLASSIFICATION_DATASET = "breast_cancer"

TARGET_COLUMN = "target"
DATASET_TASKS = {
    "diabetes": "regression",
    "california_housing": "regression",
    "breast_cancer": "classification",
}


# --------------------------------------------------------------------------
# 1. Loading datasets
# --------------------------------------------------------------------------
def _fetch_from_sklearn(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load one of the supported datasets straight from scikit-learn."""
    if name == "diabetes":
        from sklearn.datasets import load_diabetes

        # scaled=False keeps the columns in their real units (age in years,
        # bmi, blood pressure...) instead of scikit-learn's pre-standardized
        # version - otherwise the StandardScaler leakage demo would be
        # scaling data that is already scaled.
        bunch = load_diabetes(as_frame=True, scaled=False)
    elif name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer

        bunch = load_breast_cancer(as_frame=True)
    else:  # california_housing
        from sklearn.datasets import fetch_california_housing

        bunch = fetch_california_housing(as_frame=True)
    return bunch.data, bunch.target


def get_dataset_task(name: str) -> str:
    """Return 'regression' or 'classification' for a supported dataset name."""
    key = name.lower() if isinstance(name, str) else name
    if key not in DATASET_TASKS:
        raise ValueError(f"Unknown dataset '{name}'. Choose one of: {sorted(DATASET_TASKS)}")
    return DATASET_TASKS[key]


def load_ml_dataset(name: str, data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a dataset as (features DataFrame X, target Series y), caching it as a CSV.

    The first call fetches the data from scikit-learn and saves it to
    `<data_dir>/<name>.csv` (columns = features + a 'target' column).
    Later calls read that CSV, so the project works offline afterwards.

    Args:
        name: 'diabetes', 'breast_cancer', or 'california_housing'.
        data_dir: Folder for the CSV cache (default: epic4_ml/data/).

    Returns:
        (X, y) - features DataFrame and target Series named 'target'.

    Raises:
        ValueError: If `name` is not a supported dataset.
    """
    get_dataset_task(name)  # validates the name (raises ValueError if unknown)
    key = name.lower()
    folder = Path(data_dir) if data_dir is not None else DATA_DIR
    csv_path = folder / f"{key}.csv"

    if csv_path.exists():
        frame = pd.read_csv(csv_path)
    else:
        X_raw, y_raw = _fetch_from_sklearn(key)
        frame = X_raw.copy()
        frame[TARGET_COLUMN] = y_raw.to_numpy()
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)
        logger.info("Saved dataset '%s' to %s", key, csv_path)

    X = frame.drop(columns=[TARGET_COLUMN])
    y = frame[TARGET_COLUMN]
    return X, y


# --------------------------------------------------------------------------
# 2. Train/test split with validation
# --------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    stratify: bool = False,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split (X, y) into train and test sets, with input validation.

    `random_state=42` makes the split reproducible - the same rows land in
    train and test every run, so scores from different days are comparable.

    `stratify=True` (use it for classification) keeps the class proportions
    the same in train and test. Without it, a random split of a small or
    imbalanced dataset can put too few rare-class rows in the test set and
    make scores misleading.

    Args:
        X: Feature DataFrame.
        y: Target Series (same length as X).
        test_size: Fraction of rows for the test set, strictly between 0 and 1.
        stratify: If True, stratify the split on `y`.
        random_state: Seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test)

    Raises:
        ValueError: On empty data, mismatched lengths, or a `test_size`
            outside (0, 1).
    """
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        raise ValueError("Cannot split empty data: X and y must contain at least one row")
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {len(y)}")
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise ValueError(f"test_size must be a number strictly between 0 and 1, got {test_size!r}")

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


def class_distribution(y: pd.Series) -> pd.Series:
    """Return each class's share of `y` as a proportion (sums to 1.0)."""
    return y.value_counts(normalize=True).sort_index()


def compare_class_distribution(y: pd.Series, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Side-by-side class proportions: full data vs. train vs. test.

    If stratification worked, the three columns are almost identical.
    """
    table = pd.DataFrame({
        "full": class_distribution(y),
        "train": class_distribution(y_train),
        "test": class_distribution(y_test),
    })
    table["max_gap_vs_full"] = table[["train", "test"]].sub(table["full"], axis=0).abs().max(axis=1)
    return table


# --------------------------------------------------------------------------
# 3. Baseline models
# --------------------------------------------------------------------------
def get_baseline_scores(X_train, X_test, y_train, y_test, task: str) -> Dict[str, Any]:
    """Score a 'dumb' baseline that ignores the features entirely.

    - regression:     DummyRegressor(strategy='mean') always predicts the
                      training-set mean. Its R^2 on test is ~0 (a bit
                      negative), so any real model must beat ~0.
    - classification: DummyClassifier(strategy='most_frequent') always
                      predicts the most common training class. Its accuracy
                      equals that class's share of the test set.

    Args:
        X_train, X_test, y_train, y_test: The split data.
        task: 'regression' or 'classification'.

    Returns:
        Dict. Common keys: 'task', 'strategy', 'primary_metric',
        'primary_score'. Regression adds train_r2, test_r2, test_rmse,
        test_mae, baseline_prediction. Classification adds train_accuracy,
        test_accuracy, test_f1, majority_class.

    Raises:
        ValueError: If `task` is not 'regression' or 'classification'.
    """
    if task == "regression":
        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        test_r2 = float(r2_score(y_test, test_pred))
        return {
            "task": task,
            "strategy": "mean",
            "primary_metric": "test_r2",
            "primary_score": test_r2,
            "train_r2": float(r2_score(y_train, model.predict(X_train))),
            "test_r2": test_r2,
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "baseline_prediction": float(model.constant_[0][0]),
        }

    if task == "classification":
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        test_accuracy = float(accuracy_score(y_test, test_pred))
        return {
            "task": task,
            "strategy": "most_frequent",
            "primary_metric": "test_accuracy",
            "primary_score": test_accuracy,
            "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
            "test_accuracy": test_accuracy,
            "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
            "majority_class": model.classes_[int(np.argmax(model.class_prior_))].item(),
        }

    raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")


# --------------------------------------------------------------------------
# 4. Data leakage demonstration
# --------------------------------------------------------------------------
def _infer_task(y: pd.Series) -> str:
    """Guess the task from the target: few integer/text classes -> classification."""
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
        return "classification"
    return "regression"


def demonstrate_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Show scaler data leakage: fit StandardScaler on ALL data vs. on TRAIN only.

    LEAKY (wrong):   scaler.fit(X_full) -> the scaler's mean/std were
                     computed using the TEST rows too. Information about
                     the test set has quietly influenced how the training
                     data is prepared, so the test set is no longer truly
                     'unseen' and the reported score is dishonest.
    CLEAN (correct): scaler.fit(X_train) -> mean/std come from training rows
                     only; the test set is transformed with those same
                     training statistics, exactly like brand-new data
                     would be in real life.

    What to expect (and why leakage is dangerous): the scaler's statistics
    differ measurably between the two setups (see 'scaler_mean_gap'), but
    the final score often changes only slightly - and not always upward.
    That is exactly the problem: the mistake is SILENT. Nothing crashes, the
    score just becomes untrustworthy, and the damage grows with small
    datasets, outliers, and fancier preprocessing (feature selection,
    imputation, PCA). Rule: every fit() - scalers included - sees train
    data only; the test set only ever gets transform()/predict().

    A KNN model is used on purpose: it is distance-based, so unlike plain
    linear regression it is genuinely affected by how features are scaled.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Test fraction.
        random_state: Seed for the split.

    Returns:
        Dict with 'task', 'model', 'metric', 'leaky_score', 'clean_score',
        'score_difference' (leaky - clean), and 'scaler_mean_gap' (the
        largest difference between the two scalers' means, in units of the
        training standard deviation).
    """
    task = _infer_task(y)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, stratify=(task == "classification"), random_state=random_state)

    if task == "classification":
        make_model, score_fn, metric = (lambda: KNeighborsClassifier(n_neighbors=5)), accuracy_score, "accuracy"
        model_name = "KNeighborsClassifier(k=5)"
    else:
        make_model, score_fn, metric = (lambda: KNeighborsRegressor(n_neighbors=5)), r2_score, "r2"
        model_name = "KNeighborsRegressor(k=5)"

    # LEAKY: the scaler sees every row, including the future test rows.
    leaky_scaler = StandardScaler().fit(X)
    leaky_model = make_model().fit(leaky_scaler.transform(X_train), y_train)
    leaky_score = float(score_fn(y_test, leaky_model.predict(leaky_scaler.transform(X_test))))

    # CLEAN: the scaler sees the training rows only.
    clean_scaler = StandardScaler().fit(X_train)
    clean_model = make_model().fit(clean_scaler.transform(X_train), y_train)
    clean_score = float(score_fn(y_test, clean_model.predict(clean_scaler.transform(X_test))))

    mean_gap = float(np.max(np.abs(leaky_scaler.mean_ - clean_scaler.mean_) / clean_scaler.scale_))

    return {
        "task": task,
        "model": model_name,
        "metric": metric,
        "leaky_score": leaky_score,
        "clean_score": clean_score,
        "score_difference": leaky_score - clean_score,
        "scaler_mean_gap": mean_gap,
    }


# --------------------------------------------------------------------------
# 5. MLExperiment - reused all week
# --------------------------------------------------------------------------
class MLExperiment:
    """Bundle one dataset's split, baseline, and (later) model scores.

    Built to be reused all week: `run()` loads + splits + scores the
    baseline once, keeps the split arrays as attributes, and later days can
    call `record_model(name, score)` to line real models up against the
    baseline in `summary()`.
    """

    def __init__(self, dataset_name: str, test_size: float = DEFAULT_TEST_SIZE, data_dir: Optional[Path] = None):
        self.dataset_name = dataset_name
        self.task = get_dataset_task(dataset_name)
        self.test_size = test_size
        self.data_dir = data_dir
        self.stratified = self.task == "classification"

        self.n_features: Optional[int] = None
        self.n_train: Optional[int] = None
        self.n_test: Optional[int] = None
        self.baseline_scores: Optional[Dict[str, Any]] = None
        self.model_scores: Dict[str, float] = {}
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def run(self) -> "MLExperiment":
        """Load the dataset, split it, and compute baseline scores. Returns self."""
        X, y = load_ml_dataset(self.dataset_name, self.data_dir)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, self.test_size, stratify=self.stratified
        )
        self.n_features = X.shape[1]
        self.n_train = len(self.X_train)
        self.n_test = len(self.X_test)
        self.baseline_scores = get_baseline_scores(
            self.X_train, self.X_test, self.y_train, self.y_test, self.task
        )
        return self

    @property
    def baseline_score(self) -> Optional[float]:
        """The baseline's headline score (test R^2 or test accuracy), or None before run()."""
        return None if self.baseline_scores is None else self.baseline_scores["primary_score"]

    def record_model(self, name: str, score: float) -> None:
        """Store a real model's score so summary() can compare it with the baseline."""
        self.model_scores[name] = float(score)

    def summary(self) -> str:
        """Return a short human-readable report of this experiment."""
        header = f"Experiment: {self.dataset_name} ({self.task})"
        if self.baseline_scores is None:
            return f"{header}\n  (not run yet - call run() first)"

        split_kind = "stratified" if self.stratified else "random"
        metric = self.baseline_scores["primary_metric"]
        strategy = self.baseline_scores["strategy"]
        lines = [
            header,
            f"  Features: {self.n_features}",
            f"  Split: {self.n_train} train / {self.n_test} test "
            f"(test_size={self.test_size}, {split_kind}, random_state={RANDOM_STATE})",
            f"  Baseline (Dummy, {strategy}): {metric} = {self.baseline_score:.4f}",
        ]
        if self.model_scores:
            lines.append("  Models:")
            for name, score in self.model_scores.items():
                lines.append(f"    {name}: {metric} = {score:.4f} ({score - self.baseline_score:+.4f} vs baseline)")
        else:
            lines.append("  Models: none recorded yet")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 6. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    import sklearn

    logger.info("=== Day 16: ML Fundamentals, Train/Test Split & Baseline ===")
    logger.info("scikit-learn version: %s", sklearn.__version__)

    experiments = {}
    for name in (REGRESSION_DATASET, CLASSIFICATION_DATASET):
        logger.info("--- Dataset: %s ---", name)
        experiment = MLExperiment(name).run()
        experiments[name] = experiment
        logger.info("Baseline scores: %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in experiment.baseline_scores.items()})

        if experiment.task == "classification":
            X, y = load_ml_dataset(name)
            distribution = compare_class_distribution(y, experiment.y_train, experiment.y_test)
            logger.info("Class distribution before (full) and after (train/test) the STRATIFIED split:\n%s", distribution.round(4))
            logger.info(
                "Largest gap between a split's class share and the full data: %.4f "
                "(stratification worked if this is tiny)", distribution["max_gap_vs_full"].max(),
            )

    logger.info("--- Data leakage demonstration (StandardScaler fit on all data vs. train only) ---")
    for name in (REGRESSION_DATASET, CLASSIFICATION_DATASET):
        X, y = load_ml_dataset(name)
        result = demonstrate_leakage(X, y)
        logger.info(
            "%s | %s %s: leaky=%.4f clean=%.4f difference=%+.4f | scaler mean gap=%.4f std-units",
            name, result["model"], result["metric"], result["leaky_score"],
            result["clean_score"], result["score_difference"], result["scaler_mean_gap"],
        )

    logger.info("--- Experiment summaries ---")
    for experiment in experiments.values():
        logger.info("\n%s", experiment.summary())


if __name__ == "__main__":
    main()
