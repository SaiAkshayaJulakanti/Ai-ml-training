"""Day 20: Decision Trees, Random Forest & Epic 4 Deliverable.

Dependency: Days 16-19 complete. This module is the Epic 4 capstone - it
pulls together every dataset, preprocessor, and model built this week into
one shared data-loading path (`load_experiment_data`) and one comparison
function (`train_and_compare_models`), the same way Day 15 pulled Epic 3's
Days 11-14 together into one report.

=============================================================================
Why trees overfit, and why a forest helps (in plain words)
=============================================================================
A decision tree with no depth limit keeps splitting until it can perfectly
separate the training data - including the training data's RANDOM NOISE,
not just the real pattern. That is overfitting: training accuracy climbs
toward 100% while test accuracy stalls or actively gets WORSE, because the
tree memorized quirks specific to the training rows rather than learning
something that generalizes. `depth_vs_score_analysis` below makes this
visible directly: as `max_depth` grows, watch the train/test score gap widen.

A Random Forest trains many trees, each on a random subset of ROWS
(bootstrap sampling) and a random subset of COLUMNS at each split, then
averages their predictions. Each individual tree still overfits its own
random slice of the data, but the trees overfit to DIFFERENT noise - when
averaged, that noise mostly cancels out while the real, shared signal
reinforces itself. This is why a forest typically generalizes better than
any single one of its own trees, at the cost of being harder to visualize
or explain as a single diagram (hence Day 20 only plots ONE small tree,
not the whole forest).
=============================================================================
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from epic4_ml.day16_ml_basics import get_baseline_scores, load_ml_dataset, split_data
    from epic4_ml.day17_feature_engineering import (
        NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS,
        add_engineered_features, build_preprocessor, get_feature_names, load_raw_telco,
    )
    from epic4_ml.day18_linear_regression import regression_metrics
    from epic4_ml.day19_logistic_regression import classification_report_dict
except ImportError:
    from day16_ml_basics import get_baseline_scores, load_ml_dataset, split_data
    from day17_feature_engineering import (
        NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS,
        add_engineered_features, build_preprocessor, get_feature_names, load_raw_telco,
    )
    from day18_linear_regression import regression_metrics
    from day19_logistic_regression import classification_report_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHARTS_DIR = Path(__file__).parent / "charts"
RANDOM_STATE = 42
REGRESSION_DATASET = "diabetes"


# --------------------------------------------------------------------------
# 0. Shared data loading - one path, reused by train_and_compare_models AND train.py
# --------------------------------------------------------------------------
def load_experiment_data(task: str) -> Dict[str, Any]:
    """Load, split, and preprocess this week's dataset for `task`, using Day 17's preprocessor.

    Args:
        task: 'classification' (Telco Churn) or 'regression' (Diabetes).

    Returns:
        Dict with 'X_train', 'X_test' (preprocessed numpy arrays),
        'y_train', 'y_test', 'feature_names', 'preprocessor' (fitted),
        and 'task'.

    Raises:
        ValueError: If `task` is not 'classification' or 'regression'.
    """
    if task == "classification":
        raw = load_raw_telco()
        y = raw["Churn"]
        X_raw = raw.drop(columns=["Churn"])
        X_train_raw, X_test_raw, y_train, y_test = split_data(
            X_raw, y, test_size=0.2, stratify=True, random_state=RANDOM_STATE
        )
        X_train_fe = add_engineered_features(X_train_raw)
        X_test_fe = add_engineered_features(X_test_raw)
        preprocessor = build_preprocessor(NUMERIC_COLS, NOMINAL_COLS, ORDINAL_COLS)
        X_train = preprocessor.fit_transform(X_train_fe, y_train)
        X_test = preprocessor.transform(X_test_fe)

    elif task == "regression":
        X, y = load_ml_dataset(REGRESSION_DATASET)
        X_train_raw, X_test_raw, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RANDOM_STATE)
        preprocessor = build_preprocessor(numeric_cols=list(X.columns), categorical_cols=[], ordinal_cols={})
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

    else:
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

    return {
        "task": task,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": get_feature_names(preprocessor),
        "preprocessor": preprocessor,
    }


# --------------------------------------------------------------------------
# 1. Depth vs. score - overfitting, made visible
# --------------------------------------------------------------------------
def depth_vs_score_analysis(
    X_train, X_test, y_train, y_test, depths: List[Optional[int]], task: str = "classification",
) -> pd.DataFrame:
    """Train a decision tree at each depth in `depths` and record train/test score.

    Uses each model's own `.score()` - accuracy for a classifier, R^2 for a
    regressor - so the same function works for either task. `None` is a
    valid depth (sklearn's "no limit", the tree grows until every leaf is
    pure or has too few samples to split).

    Args:
        X_train, X_test, y_train, y_test: The split, preprocessed data.
        depths: Depths to try, e.g. [2, 4, 6, 10, None].
        task: 'classification' or 'regression'.

    Returns:
        DataFrame with one row per depth: columns 'max_depth' (as a string,
        so `None` displays cleanly), 'train_score', 'test_score',
        'overfit_gap' (train_score - test_score; a widening gap as depth
        grows IS overfitting).

    Raises:
        ValueError: If `depths` is empty or `task` is invalid.
    """
    if not depths:
        raise ValueError("depths must contain at least one value")
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

    ModelClass = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor

    rows = []
    for depth in depths:
        model = ModelClass(max_depth=depth, random_state=RANDOM_STATE).fit(X_train, y_train)
        train_score = float(model.score(X_train, y_train))
        test_score = float(model.score(X_test, y_test))
        rows.append({
            "max_depth": str(depth),
            "train_score": train_score,
            "test_score": test_score,
            "overfit_gap": train_score - test_score,
        })

    return pd.DataFrame(rows)


def plot_depth_vs_score(df: pd.DataFrame, save_path: str, task: str = "classification") -> None:
    """Plot train vs. test score across depths, saved as a .png.

    A healthy region is where both lines track closely together; the point
    where they start to diverge is where the tree begins overfitting.
    """
    metric_label = "Accuracy" if task == "classification" else "R\u00b2"
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    ax.plot(x, df["train_score"], marker="o", color="#4C72B0", linewidth=2, label="Train score")
    ax.plot(x, df["test_score"], marker="o", color="#C44E52", linewidth=2, label="Test score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["max_depth"])
    ax.set_xlabel("max_depth")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Decision Tree: Train vs. Test {metric_label} by Depth (Overfitting Check)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved depth-vs-score plot: %s", save_path)


# --------------------------------------------------------------------------
# 2. Tree visualization
# --------------------------------------------------------------------------
def plot_tree_diagram(
    model: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    feature_names: List[str],
    save_path: str,
    class_names: Optional[List[str]] = None,
) -> None:
    """Visualize a FITTED, shallow decision tree with sklearn's plot_tree, saved as a .png.

    Only meaningful for a shallow tree (max_depth around 2-4) - anything
    deeper becomes an unreadable wall of boxes, which is itself part of why
    a single tree's interpretability advantage fades quickly as depth grows.

    Args:
        model: A fitted DecisionTreeClassifier or DecisionTreeRegressor.
        feature_names: Names matching the model's input columns, in order.
        save_path: Where to save the .png.
        class_names: For a classifier, display names for each class
            (e.g. ["No churn", "Churn"]). Ignored for a regressor.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        model, feature_names=feature_names, class_names=class_names,
        filled=True, rounded=True, fontsize=9, ax=ax,
    )
    ax.set_title(f"Decision Tree (max_depth={model.get_params()['max_depth']})")
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved tree diagram: %s", save_path)


# --------------------------------------------------------------------------
# 3. Feature importance
# --------------------------------------------------------------------------
def plot_feature_importance(model: Any, feature_names: List[str], save_path: str, top_n: int = 10) -> None:
    """Plot the top-`n` features by `model.feature_importances_` as a horizontal bar chart.

    Random Forest (and any tree-based model) scores each feature by how
    much, on average across all its trees/splits, that feature reduced
    impurity (classification) or variance (regression) when used to split
    - higher means the model leaned on that feature more. Importances
    across ALL features always sum to 1.0, so a feature's importance is
    naturally interpretable as "this feature accounted for X% of the
    model's total splitting power."

    Args:
        model: A fitted model with a `feature_importances_` attribute
            (e.g. RandomForestClassifier/Regressor, DecisionTree*).
        feature_names: Names matching the model's input columns, in order.
        save_path: Where to save the .png.
        top_n: How many top features to show.

    Raises:
        AttributeError: If `model` has no `feature_importances_` attribute.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{type(model).__name__} has no feature_importances_ - is it fitted and tree-based?")

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    top_features = importances.head(top_n).iloc[::-1]  # reversed so the biggest bar ends up on top

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(top_features))))
    ax.barh(top_features.index, top_features.values, color="#55A868", edgecolor="black")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(top_features)} Feature Importances ({type(model).__name__})")
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance plot: %s", save_path)


# --------------------------------------------------------------------------
# 4. Model comparison across the whole week
# --------------------------------------------------------------------------
def _build_model(task: str, name: str) -> Any:
    """Construct (but do not fit) a model by name, for either task."""
    if task == "classification":
        registry = {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "logistic_regression_balanced": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
            "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
        }
    else:
        registry = {
            "dummy": DummyRegressor(strategy="mean"),
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
            "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
        }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}' for task '{task}'. Choose one of: {sorted(registry)}")
    return registry[name]


def train_and_compare_models(task: str) -> pd.DataFrame:
    """Train every model from this week on `task`'s dataset and return one comparison table.

    Classification models: Dummy baseline, LogisticRegression,
    LogisticRegression(balanced), DecisionTreeClassifier, RandomForestClassifier.
    Regression models: Dummy baseline, LinearRegression,
    DecisionTreeRegressor, RandomForestRegressor.

    All models train on the IDENTICAL preprocessed train/test split from
    `load_experiment_data`, so the comparison is apples-to-apples.

    Args:
        task: 'classification' or 'regression'.

    Returns:
        DataFrame, one row per model, with task-appropriate metric columns
        (accuracy/precision/recall/f1/roc_auc for classification;
        mae/rmse/r2 for regression).
    """
    data = load_experiment_data(task)
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    model_names = (
        ["dummy", "logistic_regression", "logistic_regression_balanced", "decision_tree", "random_forest"]
        if task == "classification" else
        ["dummy", "linear_regression", "decision_tree", "random_forest"]
    )

    rows = []
    for name in model_names:
        model = _build_model(task, name).fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task == "classification":
            y_prob = model.predict_proba(X_test)[:, 1]
            report = classification_report_dict(y_test, y_pred, y_prob)
            rows.append({
                "model": name, "accuracy": round(report["accuracy"], 4),
                "precision": round(report["precision"], 4), "recall": round(report["recall"], 4),
                "f1": round(report["f1"], 4), "roc_auc": round(report["roc_auc"], 4),
            })
        else:
            report = regression_metrics(y_test, y_pred)
            rows.append({
                "model": name, "mae": round(report["mae"], 3),
                "rmse": round(report["rmse"], 3), "r2": round(report["r2"], 4),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Persistence - model + metrics bundle
# --------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Recursively convert numpy types (arrays, scalars) into plain JSON-serializable Python types."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_model_bundle(model: Any, metrics: Dict[str, Any], output_dir: str) -> Tuple[Path, Path]:
    """Save a fitted model and its metrics as a reusable bundle: model.joblib + metrics.json.

    Args:
        model: A fitted model (or Pipeline) to persist with joblib.
        metrics: A metrics dict (e.g. from `classification_report_dict` or
            `regression_metrics`) - numpy types inside it are converted to
            plain Python types so the file is valid JSON.
        output_dir: Directory to write both files into (created if needed).

    Returns:
        (model_path, metrics_path) - the two files written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "model.joblib"
    metrics_path = out / "metrics.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2)

    logger.info("Saved model bundle: %s, %s", model_path, metrics_path)
    return model_path, metrics_path


# --------------------------------------------------------------------------
# 6. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 20: Decision Trees, Random Forest & Epic 4 Deliverable ===")

    for task in ("classification", "regression"):
        logger.info("--- Task: %s ---", task)
        data = load_experiment_data(task)
        X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
        feature_names = data["feature_names"]

        # --- Depth vs. score: overfitting, made visible ---
        depth_df = depth_vs_score_analysis(X_train, X_test, y_train, y_test, depths=[2, 4, 6, 10, None], task=task)
        logger.info("Depth vs. score:\n%s", depth_df.to_string(index=False))
        plot_depth_vs_score(depth_df, str(CHARTS_DIR / f"day20_{task}_depth_vs_score.png"), task=task)

        # --- One small, readable tree ---
        TreeClass = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
        small_tree = TreeClass(max_depth=3, random_state=RANDOM_STATE).fit(X_train, y_train)
        class_names = ["No churn", "Churn"] if task == "classification" else None
        plot_tree_diagram(small_tree, feature_names, str(CHARTS_DIR / f"day20_{task}_tree.png"), class_names=class_names)

        # --- Random forest + feature importance ---
        ForestClass = RandomForestClassifier if task == "classification" else RandomForestRegressor
        forest = ForestClass(n_estimators=200, max_depth=10, random_state=RANDOM_STATE).fit(X_train, y_train)
        logger.info("Feature importances sum to %.4f (should be ~1.0)", forest.feature_importances_.sum())
        plot_feature_importance(forest, feature_names, str(CHARTS_DIR / f"day20_{task}_feature_importance.png"))

        # --- Full model comparison ---
        comparison = train_and_compare_models(task)
        logger.info("Model comparison (%s):\n%s", task, comparison.to_string(index=False))

    logger.info("=== Day 20 complete - see epic4_ml/README.md for the final write-up ===")


if __name__ == "__main__":
    main()
