"""Day 11: Descriptive Statistics & Probability Fundamentals - Epic 3.

Dependency: reads the cleaned/merged e-commerce dataset produced by the
Epic 2 checkpoint (Day 7 cleaning + Day 8 merge, re-derived the same way
Day 9 does it - see `_load_merged_dataset`).

This module builds a statistical foundation from first principles:
- Central tendency and spread are implemented from scratch with NumPy
  (no `.describe()` shortcut) and cross-validated against `scipy.stats`/
  `pandas` equivalents.
- Basic probability primitives (simple, conditional, Bayes' theorem) are
  implemented and demonstrated on a small worked email-spam dataset, then
  applied to the real e-commerce dataset via boolean indicator columns.

Population vs. sample variance/std (see `compute_spread`):
    - Population variance/std (ddof=0) is used when `data` IS the entire
      population you care about - every value that exists is in `data`.
      Formula: var = sum((x - mean)**2) / n
    - Sample variance/std (ddof=1, "Bessel's correction") is used when
      `data` is a sample drawn from a larger population and you want an
      unbiased estimate of the population variance.
      Formula: var = sum((x - mean)**2) / (n - 1)
      Dividing by (n - 1) instead of n corrects for the fact that a
      sample's own mean is closer to its data points than the true
      population mean would be, which would otherwise make the naive
      (ddof=0) estimator systematically underestimate population variance.
    - NumPy's `np.var`/`np.std` default to ddof=0 (population).
      pandas' `.var()`/`.std()` default to ddof=1 (sample).
    - In this module: the e-commerce dataset (150 orders) is treated as a
      SAMPLE of a larger ongoing order stream, so ddof=1 is the
      scientifically appropriate choice for inference - but both are
      computed and reported so the difference is visible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Epic 2 checkpoint dataset (Day 7 cleaned orders + Day 8 category lookup),
# reused here rather than duplicated - see day9_eda.py for the original.
EPIC2_DATA_DIR = Path(__file__).parent.parent / "epic2_pandas" / "data"
CLEANED_CSV = EPIC2_DATA_DIR / "cleaned_dataset.csv"
CATEGORY_LOOKUP_CSV = EPIC2_DATA_DIR / "category_lookup.csv"


def _load_merged_dataset() -> pd.DataFrame:
    """Load and merge the Epic 2 cleaned orders with the category lookup.

    Rebuilt fresh via the same left join Day 8/Day 9 demonstrate, so this
    module has no hidden dependency on an intermediate saved file.
    """
    orders = pd.read_csv(CLEANED_CSV)
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    lookup = pd.read_csv(CATEGORY_LOOKUP_CSV)
    return pd.merge(orders, lookup, on="product_category", how="left")


def _percentile_from_scratch(sorted_data: np.ndarray, q: float) -> float:
    """Compute the q-th percentile (0-100) of already-sorted data by hand.

    Uses the same linear-interpolation-between-closest-ranks method NumPy's
    `np.percentile` (and pandas' `.quantile`) use by default, so results
    match those libraries within floating-point tolerance rather than
    merely being "a" reasonable percentile definition among several.
    """
    n = len(sorted_data)
    if n == 0:
        raise ValueError("Cannot compute a percentile of empty data")
    if n == 1:
        return float(sorted_data[0])

    rank = (q / 100.0) * (n - 1)
    lower_idx = int(np.floor(rank))
    upper_idx = int(np.ceil(rank))
    if lower_idx == upper_idx:
        return float(sorted_data[lower_idx])

    fraction = rank - lower_idx
    return float(
        sorted_data[lower_idx] + fraction * (sorted_data[upper_idx] - sorted_data[lower_idx])
    )


# --------------------------------------------------------------------------
# 1. Central tendency - from scratch
# --------------------------------------------------------------------------
def compute_central_tendency(data: np.ndarray) -> Dict[str, float]:
    """Compute mean, median, and mode of `data` from first principles.

    - mean: sum of values / count of values.
    - median: the 50th percentile via linear interpolation on sorted data
      (equivalent to averaging the two middle values for an even-length
      array, matching `np.median`).
    - mode: the most frequently occurring value, found by counting
      occurrences of each unique value (ties broken by the smallest value,
      matching `scipy.stats.mode`'s default tie-breaking behavior).

    Args:
        data: 1-D array of numeric values.

    Returns:
        Dict with keys 'mean', 'median', 'mode'.

    Raises:
        ValueError: If `data` is empty.
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    if n == 0:
        raise ValueError("Cannot compute central tendency of empty data")

    mean = float(np.sum(data) / n)

    sorted_data = np.sort(data)
    median = _percentile_from_scratch(sorted_data, 50)

    unique_values, counts = np.unique(data, return_counts=True)
    max_count = counts.max()
    # np.unique returns unique_values already sorted ascending, so taking
    # the first tied value naturally matches scipy.stats.mode's convention
    # of returning the smallest mode on a tie.
    mode = float(unique_values[counts == max_count][0])

    return {"mean": mean, "median": median, "mode": mode}


# --------------------------------------------------------------------------
# 2. Spread - from scratch
# --------------------------------------------------------------------------
def compute_spread(data: np.ndarray) -> Dict[str, float]:
    """Compute variance, standard deviation, range, and IQR of `data`.

    Both population (ddof=0) and sample (ddof=1) variance/std are
    reported - see the module docstring for when to use which. `range`
    and `iqr` have no population-vs-sample distinction.

    Args:
        data: 1-D array of numeric values.

    Returns:
        Dict with keys: 'variance_population', 'variance_sample',
        'std_population', 'std_sample', 'range', 'iqr'.

    Raises:
        ValueError: If `data` has fewer than 2 elements (sample variance
            is undefined for n < 2, since it divides by n - 1).
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    if n < 2:
        raise ValueError("compute_spread requires at least 2 data points (sample variance divides by n - 1)")

    mean = float(np.sum(data) / n)
    squared_deviations = np.sum((data - mean) ** 2)

    variance_population = float(squared_deviations / n)
    variance_sample = float(squared_deviations / (n - 1))

    sorted_data = np.sort(data)
    q1 = _percentile_from_scratch(sorted_data, 25)
    q3 = _percentile_from_scratch(sorted_data, 75)

    return {
        "variance_population": variance_population,
        "variance_sample": variance_sample,
        "std_population": float(np.sqrt(variance_population)),
        "std_sample": float(np.sqrt(variance_sample)),
        "range": float(sorted_data[-1] - sorted_data[0]),
        "iqr": q3 - q1,
    }


# --------------------------------------------------------------------------
# 3. Probability - from scratch
# --------------------------------------------------------------------------
def simple_probability(df: pd.DataFrame, event: str) -> float:
    """Compute P(event) as the fraction of rows where boolean column `event` is True.

    Args:
        df: Input DataFrame.
        event: Name of a boolean (or 0/1) indicator column.

    Returns:
        P(event) as a float in [0, 1].

    Raises:
        ValueError: If `event` is not a column in `df`.
    """
    if event not in df.columns:
        raise ValueError(f"Column '{event}' not found in DataFrame")
    return float(df[event].astype(bool).mean())


def conditional_probability(df: pd.DataFrame, event_a: str, event_b: str) -> float:
    """Compute P(A | B) = P(A and B) / P(B) from boolean indicator columns.

    Args:
        df: Input DataFrame.
        event_a: Name of the boolean indicator column for event A.
        event_b: Name of the boolean indicator column for event B
            (the event being conditioned on).

    Returns:
        P(event_a | event_b) as a float in [0, 1].

    Raises:
        ValueError: If either column is missing, or if P(event_b) == 0
            (conditioning on an impossible event is undefined).
    """
    for col in (event_a, event_b):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    mask_b = df[event_b].astype(bool)
    p_b = float(mask_b.mean())
    if p_b == 0:
        raise ValueError(f"P({event_b}) = 0; conditional probability is undefined")

    mask_a_and_b = df[event_a].astype(bool) & mask_b
    p_a_and_b = float(mask_a_and_b.mean())
    return p_a_and_b / p_b


def bayes_theorem(p_a: float, p_b_given_a: float, p_b: float) -> float:
    """Compute P(A | B) via Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B).

    Args:
        p_a: Prior probability P(A).
        p_b_given_a: Likelihood P(B | A).
        p_b: Marginal probability P(B) (the normalizing "evidence" term).

    Returns:
        The posterior probability P(A | B).

    Raises:
        ValueError: If any probability is outside [0, 1], or if p_b == 0.
    """
    for name, value in (("p_a", p_a), ("p_b_given_a", p_b_given_a), ("p_b", p_b)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a probability in [0, 1], got {value}")
    if p_b == 0:
        raise ValueError("p_b = 0; Bayes' theorem is undefined when the evidence has zero probability")

    return (p_b_given_a * p_a) / p_b


# --------------------------------------------------------------------------
# 4. Worked example - email spam classification (Bayes' theorem)
# --------------------------------------------------------------------------
def spam_bayes_worked_example() -> Dict[str, float]:
    """Run a worked Bayes' theorem example on a small spam-classification dataset.

    Scenario (counts from a hypothetical inbox of 200 emails):
        - 40 emails are spam, 160 are ham (not spam).
        - Of the 40 spam emails, 32 contain the word "free".
        - Of the 160 ham emails, 8 contain the word "free".

    Question: given that an email contains the word "free", what is the
    probability it is spam?  P(spam | "free") via Bayes' theorem.

    Returns:
        Dict with the input rates and the computed posterior
        'p_spam_given_free'.
    """
    total_emails = 200
    spam_count = 40
    ham_count = total_emails - spam_count

    spam_with_free = 32
    ham_with_free = 8

    p_spam = spam_count / total_emails
    p_ham = ham_count / total_emails
    p_free_given_spam = spam_with_free / spam_count
    p_free_given_ham = ham_with_free / ham_count

    # Law of total probability: P(free) = P(free|spam)P(spam) + P(free|ham)P(ham)
    p_free = p_free_given_spam * p_spam + p_free_given_ham * p_ham

    p_spam_given_free = bayes_theorem(p_a=p_spam, p_b_given_a=p_free_given_spam, p_b=p_free)

    return {
        "p_spam": p_spam,
        "p_ham": p_ham,
        "p_free_given_spam": p_free_given_spam,
        "p_free_given_ham": p_free_given_ham,
        "p_free": p_free,
        "p_spam_given_free": p_spam_given_free,
    }


# --------------------------------------------------------------------------
# 5. Cross-validation against scipy/pandas
# --------------------------------------------------------------------------
def _scipy_mode(data: np.ndarray) -> float:
    """Return scipy.stats.mode's modal value, handling API differences across scipy versions."""
    try:
        result = scipy_stats.mode(data, keepdims=True)  # scipy >= 1.9
    except TypeError:
        result = scipy_stats.mode(data)  # older scipy
    return float(np.ravel(result.mode)[0])


def cross_validate_column(data: np.ndarray, column_name: str = "") -> Dict[str, Dict[str, float]]:
    """Compute stats both from-scratch and via scipy/pandas, for side-by-side comparison.

    Args:
        data: 1-D array of numeric values.
        column_name: Optional label, used only in logging.

    Returns:
        Dict with 'from_scratch' and 'library' sub-dicts sharing the same
        keys, suitable for printing as a comparison table.
    """
    data = np.asarray(data, dtype=float)
    series = pd.Series(data)

    scratch_ct = compute_central_tendency(data)
    scratch_spread = compute_spread(data)

    library = {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "mode": _scipy_mode(data),
        "variance_population": float(np.var(data, ddof=0)),
        "variance_sample": float(series.var(ddof=1)),
        "std_population": float(np.std(data, ddof=0)),
        "std_sample": float(series.std(ddof=1)),
        "range": float(np.ptp(data)),
        "iqr": float(scipy_stats.iqr(data)),
    }

    from_scratch = {**scratch_ct, **scratch_spread}

    if column_name:
        logger.info("Cross-validation for column '%s':", column_name)
        for key in from_scratch:
            scratch_val = from_scratch[key]
            lib_val = library[key]
            match = "OK" if np.isclose(scratch_val, lib_val, atol=1e-6) else "MISMATCH"
            logger.info("  %-20s from_scratch=%-15.6f library=%-15.6f [%s]", key, scratch_val, lib_val, match)

    return {"from_scratch": from_scratch, "library": library}


# --------------------------------------------------------------------------
# 6. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 11: Descriptive Statistics & Probability Fundamentals ===")

    df = _load_merged_dataset()
    logger.info("Loaded merged Epic 2 dataset: %s", df.shape)

    numeric_cols = ["quantity", "unit_price"]
    comparisons: Dict[str, Any] = {}
    for col in numeric_cols:
        comparisons[col] = cross_validate_column(df[col].to_numpy(), column_name=col)

    logger.info("--- Worked Bayes' theorem example: email spam classification ---")
    spam_result = spam_bayes_worked_example()
    logger.info(
        "P(spam)=%.3f, P('free'|spam)=%.3f, P('free'|ham)=%.3f, P('free')=%.3f",
        spam_result["p_spam"], spam_result["p_free_given_spam"],
        spam_result["p_free_given_ham"], spam_result["p_free"],
    )
    logger.info(
        "=> P(spam | 'free') = %.4f: given an email contains the word 'free', "
        "there is roughly a %.0f%% chance it is spam.",
        spam_result["p_spam_given_free"], spam_result["p_spam_given_free"] * 100,
    )

    logger.info("--- Probability scenario on the e-commerce dataset ---")
    median_price = df["unit_price"].median()
    df = df.copy()
    df["is_high_value"] = df["unit_price"] > median_price
    target_category = "Electronics"
    df[f"is_category_{target_category}"] = df["product_category"] == target_category

    p_high_value = simple_probability(df, "is_high_value")
    p_category = simple_probability(df, f"is_category_{target_category}")
    p_high_value_given_category = conditional_probability(df, "is_high_value", f"is_category_{target_category}")

    logger.info("P(high-value order) = %.4f (threshold: unit_price > median $%.2f)", p_high_value, median_price)
    logger.info("P(category = %s) = %.4f", target_category, p_category)
    logger.info(
        "P(high-value | category = %s) = %.4f -- vs. the unconditional P(high-value) = %.4f",
        target_category, p_high_value_given_category, p_high_value,
    )
    if p_high_value_given_category > p_high_value:
        logger.info(
            "Interpretation: %s orders are MORE likely than average to be high-value.",
            target_category,
        )
    else:
        logger.info(
            "Interpretation: %s orders are NOT more likely than average to be high-value.",
            target_category,
        )


if __name__ == "__main__":
    main()
