#!/usr/bin/env python3
"""train.py - Epic 4 command-line training entry point.

Trains one model on one of this week's datasets, evaluates it on the held-out
test split, and saves a reusable bundle (model.joblib + metrics.json) to
`--output-dir`. This is the "ship it" step: everything explored interactively
in day16-day20 is reachable here as a single repeatable command.

Usage:
    python train.py --task classification --model random_forest --output-dir models/
    python train.py --task regression --model linear_regression --output-dir models/

Run `python train.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from epic4_ml.day19_logistic_regression import classification_report_dict
    from epic4_ml.day18_linear_regression import regression_metrics
    from epic4_ml.day20_trees_and_project import _build_model, load_experiment_data, save_model_bundle
except ImportError:
    from day19_logistic_regression import classification_report_dict
    from day18_linear_regression import regression_metrics
    from day20_trees_and_project import _build_model, load_experiment_data, save_model_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASSIFICATION_MODELS = ["dummy", "logistic_regression", "logistic_regression_balanced", "decision_tree", "random_forest"]
REGRESSION_MODELS = ["dummy", "linear_regression", "decision_tree", "random_forest"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save one of this week's models on the classification or regression dataset.",
    )
    parser.add_argument(
        "--task", required=True, choices=["classification", "regression"],
        help="Which dataset/task to train on: classification (Telco Churn) or regression (Diabetes).",
    )
    parser.add_argument(
        "--model", required=True,
        help=f"Which model to train. Classification: {CLASSIFICATION_MODELS}. Regression: {REGRESSION_MODELS}.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write model.joblib and metrics.json into (created if it doesn't exist).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    valid_models = CLASSIFICATION_MODELS if args.task == "classification" else REGRESSION_MODELS
    if args.model not in valid_models:
        logger.error("Invalid --model '%s' for --task '%s'. Choose one of: %s", args.model, args.task, valid_models)
        return 2

    logger.info("Loading and preprocessing '%s' data...", args.task)
    data = load_experiment_data(args.task)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    logger.info("Training model '%s'...", args.model)
    start = time.time()
    model = _build_model(args.task, args.model).fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info("Trained in %.2fs", elapsed)

    y_pred = model.predict(X_test)
    if args.task == "classification":
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = classification_report_dict(y_test, y_pred, y_prob)
    else:
        metrics = regression_metrics(y_test, y_pred)

    metrics["task"] = args.task
    metrics["model"] = args.model
    metrics["n_train"] = int(X_train.shape[0])
    metrics["n_test"] = int(X_test.shape[0])
    metrics["n_features"] = int(X_train.shape[1])
    metrics["train_seconds"] = round(elapsed, 4)

    model_path, metrics_path = save_model_bundle(model, metrics, args.output_dir)
    logger.info("Done. Model: %s | Metrics: %s", model_path, metrics_path)

    printable = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    logger.info("Metrics: %s", printable)
    return 0


if __name__ == "__main__":
    sys.exit(main())
