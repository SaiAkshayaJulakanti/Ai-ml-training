"""Day 12: Correlation & Distribution Concepts - Epic 3.

Dependency: reuses Day 11's stats utilities (`compute_central_tendency`,
`compute_spread`, and the Epic 2 dataset loader) rather than
re-implementing mean/std logic here - z-scores in particular are built
directly on top of `compute_spread`'s from-scratch std.

This module covers:
- Pearson correlation, implemented from scratch and cross-validated
  against `np.corrcoef` / `scipy.stats.pearsonr`.
- Spearman rank correlation via `scipy.stats.spearmanr` (used directly,
  not reimplemented - see `correlation_matrix_report` for why Spearman
  is worth having alongside Pearson).
- Generating and summarizing samples from Normal / Binomial / Poisson
  distributions.
- A normality check (Shapiro-Wilk) with p-value interpretation.
- Z-scores and threshold-based outlier flagging (±2 / ±3 std).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from day11_descriptive_stats import (
    _load_merged_dataset,
    compute_central_tendency,
    compute_spread,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Columns whose name ends in "_id" are identifiers, not measurements -
# correlating against an arbitrary sequential ID number is meaningless
# (and would just add noise to the "strongest/weakest pairs" story), so
# they're excluded from the correlation report by default.
ID_COLUMN_SUFFIX = "_id"


# --------------------------------------------------------------------------
# 1. Pearson correlation - from scratch
# --------------------------------------------------------------------------
def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Pearson correlation coefficient between `x` and `y` from scratch.

    r = sum((x - mean_x) * (y - mean_y)) / sqrt(sum((x - mean_x)^2) * sum((y - mean_y)^2))

    This is the covariance of x and y, normalized by the product of their
    standard deviations - it measures the strength and direction of a
    LINEAR relationship only. Note ddof cancels out of this formula
    entirely (it appears in both numerator-adjacent covariance and the
    two variances, and divides out), so there's no population-vs-sample
    choice to make here, unlike `compute_spread`.

    Args:
        x: 1-D array of numeric values.
        y: 1-D array of numeric values, same length as `x`.

    Returns:
        Pearson's r, in [-1, 1].

    Raises:
        ValueError: If `x` and `y` have different lengths, fewer than 2
            points, or either has zero variance (a constant column has
            no linear relationship to define).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must be the same shape, got {x.shape} and {y.shape}")
    if x.size < 2:
        raise ValueError("pearson_correlation requires at least 2 data points")

    mean_x = np.sum(x) / x.size
    mean_y = np.sum(y) / y.size
    dev_x = x - mean_x
    dev_y = y - mean_y

    numerator = np.sum(dev_x * dev_y)
    denominator = np.sqrt(np.sum(dev_x ** 2) * np.sum(dev_y ** 2))
    if denominator == 0:
        raise ValueError("Pearson correlation is undefined when x or y is constant (zero variance)")

    return float(numerator / denominator)


# --------------------------------------------------------------------------
# 2. Full pairwise correlation report - Pearson + Spearman
# --------------------------------------------------------------------------
def correlation_matrix_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson AND Spearman correlation for all numeric columns.

    Pearson vs. Spearman - when to prefer which:
        Pearson measures LINEAR association and is sensitive to outliers
        (a single extreme point can swing it a lot, since it works on
        raw values and squared deviations). Spearman instead correlates
        the RANKS of the values, so it captures any MONOTONIC relationship
        (linear or not - e.g. y = x^3) and is far more robust to outliers,
        at the cost of not distinguishing "linear" from "just consistently
        increasing." Prefer Spearman when the data is skewed, has extreme
        outliers, is ordinal, or when a monotonic-but-curved relationship
        is plausible; prefer Pearson when a straight-line relationship
        is the actual thing being modeled (e.g. before fitting a linear
        regression) and the data is reasonably well-behaved.

    One row per unique pair of numeric columns (no self-pairs, no
    duplicate A-B/B-A rows), sorted by |Pearson r| descending so the
    strongest relationships surface first.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: column_a, column_b, pearson, spearman,
        spearman_pvalue, abs_pearson.
    """
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if not col.lower().endswith(ID_COLUMN_SUFFIX)
    ]

    rows = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col_a, col_b = numeric_cols[i], numeric_cols[j]
            x = df[col_a].to_numpy(dtype=float)
            y = df[col_b].to_numpy(dtype=float)

            pearson_r = pearson_correlation(x, y)
            spearman_r, spearman_p = scipy_stats.spearmanr(x, y)

            rows.append({
                "column_a": col_a,
                "column_b": col_b,
                "pearson": pearson_r,
                "spearman": float(spearman_r),
                "spearman_pvalue": float(spearman_p),
            })

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["abs_pearson"] = report["pearson"].abs()
    return report.sort_values("abs_pearson", ascending=False).reset_index(drop=True)


def strongest_and_weakest_pairs(report: pd.DataFrame, n: int = 2) -> Dict[str, pd.DataFrame]:
    """Pull the `n` strongest and `n` weakest correlated pairs out of a correlation report.

    Args:
        report: Output of `correlation_matrix_report` (must be sorted by
            'abs_pearson' descending, which that function guarantees).
        n: How many pairs to pull from each end.

    Returns:
        Dict with 'strongest' and 'weakest' DataFrames (each up to `n` rows).
    """
    return {
        "strongest": report.head(n).reset_index(drop=True),
        "weakest": report.tail(n).reset_index(drop=True),
    }


# --------------------------------------------------------------------------
# 3. Distribution generation - Normal / Binomial / Poisson
# --------------------------------------------------------------------------
def generate_distribution_samples(size: int = 2000, seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """Generate samples from Normal, Binomial, and Poisson distributions and summarize each.

    For each distribution, the empirical (sample) mean/variance is
    compared against the theoretical mean/variance the distribution's
    parameters predict - this is the core sanity check that np.random's
    generators are behaving as documented, and builds intuition for how
    much a *finite* sample naturally wobbles around its true parameters.

    Args:
        size: Number of samples to draw from each distribution.
        seed: Random seed, for reproducibility.

    Returns:
        Dict keyed by distribution name ('normal', 'binomial', 'poisson'),
        each a dict with 'samples', 'theoretical_mean', 'theoretical_variance',
        'empirical_mean', 'empirical_variance'.
    """
    rng = np.random.default_rng(seed)

    normal_loc, normal_scale = 50.0, 10.0
    normal_samples = rng.normal(loc=normal_loc, scale=normal_scale, size=size)

    binomial_n, binomial_p = 20, 0.3
    binomial_samples = rng.binomial(n=binomial_n, p=binomial_p, size=size)

    poisson_lam = 4.0
    poisson_samples = rng.poisson(lam=poisson_lam, size=size)

    return {
        "normal": {
            "samples": normal_samples,
            "theoretical_mean": normal_loc,
            "theoretical_variance": normal_scale ** 2,
            "empirical_mean": float(np.mean(normal_samples)),
            "empirical_variance": float(np.var(normal_samples, ddof=0)),
        },
        "binomial": {
            "samples": binomial_samples,
            "theoretical_mean": binomial_n * binomial_p,
            "theoretical_variance": binomial_n * binomial_p * (1 - binomial_p),
            "empirical_mean": float(np.mean(binomial_samples)),
            "empirical_variance": float(np.var(binomial_samples, ddof=0)),
        },
        "poisson": {
            "samples": poisson_samples,
            "theoretical_mean": poisson_lam,
            "theoretical_variance": poisson_lam,  # Poisson: variance == mean == lambda
            "empirical_mean": float(np.mean(poisson_samples)),
            "empirical_variance": float(np.var(poisson_samples, ddof=0)),
        },
    }


# --------------------------------------------------------------------------
# 4. Normality check - Shapiro-Wilk
# --------------------------------------------------------------------------
def fit_normal_distribution(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Fit a Normal distribution to `data` and test whether it's plausibly normal.

    The fitted mean/std come from Day 11's own `compute_central_tendency`/
    `compute_spread` (population std, i.e. the maximum-likelihood fit -
    the same estimator `scipy.stats.norm.fit` uses under the hood),
    keeping this module built on top of Day 11 rather than re-deriving
    the same numbers a second way.

    The Shapiro-Wilk test's null hypothesis IS that the data come from a
    normal distribution:
        - p-value <= alpha: reject the null -> data is NOT well-described
          as normal (statistically significant departure).
        - p-value >  alpha: fail to reject the null -> no significant
          evidence against normality (this does NOT prove it IS normal,
          only that this test found no strong reason to doubt it).

    Args:
        data: 1-D array of numeric values.
        alpha: Significance threshold (default 0.05, i.e. 5%).

    Returns:
        Dict with 'mean', 'std', 'shapiro_statistic', 'shapiro_pvalue',
        'alpha', 'is_normal' (bool), and 'interpretation' (str).
    """
    data = np.asarray(data, dtype=float)
    mean = compute_central_tendency(data)["mean"]
    std = compute_spread(data)["std_population"]

    shapiro_stat, shapiro_p = scipy_stats.shapiro(data)
    is_normal = bool(shapiro_p > alpha)

    interpretation = (
        f"p-value ({shapiro_p:.4g}) > alpha ({alpha}): fail to reject normality - "
        f"no strong evidence this data departs from a Normal distribution."
        if is_normal else
        f"p-value ({shapiro_p:.4g}) <= alpha ({alpha}): reject normality - "
        f"this data significantly departs from a Normal distribution."
    )

    return {
        "mean": mean,
        "std": std,
        "shapiro_statistic": float(shapiro_stat),
        "shapiro_pvalue": float(shapiro_p),
        "alpha": alpha,
        "is_normal": is_normal,
        "interpretation": interpretation,
    }


# --------------------------------------------------------------------------
# 5. Z-scores and outlier flagging
# --------------------------------------------------------------------------
def compute_zscores(data: np.ndarray) -> np.ndarray:
    """Compute the z-score of every value in `data`: z = (x - mean) / std.

    Uses Day 11's `compute_central_tendency`/`compute_spread` (population
    std) as the source of mean/std, rather than recomputing them here.

    Args:
        data: 1-D array of numeric values.

    Returns:
        Array of z-scores, same shape as `data`.

    Raises:
        ValueError: If `data` has zero standard deviation (all values
            identical - z-scores are undefined, division by zero).
    """
    data = np.asarray(data, dtype=float)
    mean = compute_central_tendency(data)["mean"]
    std = compute_spread(data)["std_population"]
    if std == 0:
        raise ValueError("compute_zscores is undefined when data has zero standard deviation")
    return (data - mean) / std


def flag_zscore_outliers(data: np.ndarray, thresholds: Iterable[float] = (2.0, 3.0)) -> Dict[str, Dict[str, Any]]:
    """Flag values in `data` whose |z-score| exceeds each threshold.

    Args:
        data: 1-D array of numeric values.
        thresholds: Z-score cutoffs to report on, e.g. (2.0, 3.0) for
            the conventional "beyond 2 std" / "beyond 3 std" checks.

    Returns:
        Dict keyed by f"beyond_{threshold}_std", each a dict with
        'count', 'values', and 'indices' of the flagged points.
    """
    data = np.asarray(data, dtype=float)
    z = compute_zscores(data)

    result = {}
    for threshold in thresholds:
        mask = np.abs(z) > threshold
        result[f"beyond_{threshold}_std"] = {
            "count": int(mask.sum()),
            "values": data[mask].tolist(),
            "indices": np.where(mask)[0].tolist(),
        }
    return result


# --------------------------------------------------------------------------
# 6. Cross-validation against numpy/scipy
# --------------------------------------------------------------------------
def cross_validate_pearson(x: np.ndarray, y: np.ndarray, label: str = "") -> Dict[str, float]:
    """Compute Pearson's r from scratch and via numpy/scipy, for side-by-side comparison."""
    scratch_r = pearson_correlation(x, y)
    numpy_r = float(np.corrcoef(x, y)[0, 1])
    scipy_r, scipy_p = scipy_stats.pearsonr(x, y)

    if label:
        match = "OK" if np.isclose(scratch_r, numpy_r, atol=1e-6) and np.isclose(scratch_r, scipy_r, atol=1e-6) else "MISMATCH"
        logger.info(
            "Pearson r for %s: from_scratch=%.6f  np.corrcoef=%.6f  scipy.pearsonr=%.6f (p=%.4g) [%s]",
            label, scratch_r, numpy_r, scipy_r, scipy_p, match,
        )

    return {"from_scratch": scratch_r, "numpy": numpy_r, "scipy": float(scipy_r), "scipy_pvalue": float(scipy_p)}


# --------------------------------------------------------------------------
# 7. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 12: Correlation & Distribution Concepts ===")

    df = _load_merged_dataset()
    logger.info("Loaded merged Epic 2 dataset: %s", df.shape)

    logger.info("--- Cross-validating Pearson correlation from scratch ---")
    cross_validate_pearson(df["quantity"].to_numpy(), df["unit_price"].to_numpy(), label="quantity vs unit_price")
    cross_validate_pearson(df["unit_price"].to_numpy(), df["standard_discount_pct"].to_numpy(), label="unit_price vs standard_discount_pct")

    logger.info("--- Full correlation matrix report (Pearson + Spearman) ---")
    report = correlation_matrix_report(df)
    for _, row in report.iterrows():
        logger.info(
            "  %-25s vs %-25s  pearson=%+.4f  spearman=%+.4f (p=%.4g)",
            row["column_a"], row["column_b"], row["pearson"], row["spearman"], row["spearman_pvalue"],
        )

    extremes = strongest_and_weakest_pairs(report, n=2)
    logger.info("Strongest 2 correlated pairs:\n%s", extremes["strongest"][["column_a", "column_b", "pearson"]])
    logger.info("Weakest 2 correlated pairs:\n%s", extremes["weakest"][["column_a", "column_b", "pearson"]])
    # Observed on the canonical 150-row dataset: every pairwise |Pearson r|
    # is below 0.1 - these are synthetically generated columns with no
    # designed relationship, so near-zero correlation everywhere is the
    # EXPECTED, correctly-detected result, not a bug in the correlation code.

    logger.info("--- Distribution samples: Normal / Binomial / Poisson ---")
    distributions = generate_distribution_samples()
    for name, info in distributions.items():
        logger.info(
            "%-9s theoretical(mean=%.3f, var=%.3f)  empirical(mean=%.3f, var=%.3f)",
            name, info["theoretical_mean"], info["theoretical_variance"],
            info["empirical_mean"], info["empirical_variance"],
        )

    logger.info("--- Normality testing (Shapiro-Wilk) on real dataset columns ---")
    for col in ["quantity", "unit_price"]:
        result = fit_normal_distribution(df[col].to_numpy())
        logger.info(
            "%-12s fitted mean=%.3f std=%.3f | shapiro W=%.4f p=%.4g -> is_normal=%s",
            col, result["mean"], result["std"], result["shapiro_statistic"],
            result["shapiro_pvalue"], result["is_normal"],
        )
        logger.info("  %s", result["interpretation"])

    logger.info("--- Z-scores and outlier flagging ---")
    for col in ["quantity", "unit_price", "standard_discount_pct"]:
        outliers = flag_zscore_outliers(df[col].to_numpy())
        logger.info(
            "%-22s beyond ±2 std: %d value(s) | beyond ±3 std: %d value(s)",
            col, outliers["beyond_2.0_std"]["count"], outliers["beyond_3.0_std"]["count"],
        )
    logger.info(
        "No column in the real 150-row dataset has values beyond ±2 std - "
        "a correctly-detected NEGATIVE result (see test_day12.py for a "
        "synthetic example that DOES trigger the flag)."
    )


if __name__ == "__main__":
    main()
