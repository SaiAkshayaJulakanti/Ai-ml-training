"""common/numpy_utils.py - Shared NumPy utilities.

This module extracts genuinely reusable logic that recurs across the
Epic 1 training days (timing, statistics, z-score normalization via
broadcasting, ranking via argsort, and simple report writing) into one
shared location, rather than each day (or the Day 5 project) reimplementing
its own copy.

Where each pattern originates:
    - `timer`              -> Day 1 (decorators)
    - `array_stats`        -> Day 2 (array inspection) + Day 4 (statistical functions)
    - `zscore_normalize`   -> Day 3 (normalize_broadcast - broadcasting)
    - `rank_by_score`      -> new here, built on Day 3's argsort-based indexing ideas
    - `write_text_report`  -> new here, generic file-writing helper
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# common/ needs to reach into the sibling epic1_numpy/ package to import
# the canonical @timer decorator from Day 1 - make sure the project root
# is on sys.path regardless of how this module is imported.
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Timing - re-exported from Day 1's @timer decorator (the canonical source;
# Day 4 also imports it from Day 1 directly, per that day's spec). Defining
# it a second time here would be exactly the duplication this module exists
# to eliminate, so we import it instead.
# --------------------------------------------------------------------------
try:
    from epic1_numpy.day1_python_core import timer
except ImportError:
    from day1_python_core import timer


# --------------------------------------------------------------------------
# Statistics (originates from Day 2's inspect_array + Day 4's statistics_demo)
# --------------------------------------------------------------------------
def array_stats(arr: np.ndarray, axis: int | None = None) -> Dict[str, Any]:
    """Compute mean, median, std, min, max for `arr`, optionally along an axis.

    Args:
        arr: The array to summarize.
        axis: Axis to reduce along. None (default) computes over the
            flattened array; axis=0 computes per-column statistics for
            a 2D array, matching how compute_subject_stats uses it.

    Returns:
        A dict with keys: mean, median, std, min, max.
    """
    return {
        "mean": np.mean(arr, axis=axis),
        "median": np.median(arr, axis=axis),
        "std": np.std(arr, axis=axis),
        "min": np.min(arr, axis=axis),
        "max": np.max(arr, axis=axis),
    }


# --------------------------------------------------------------------------
# Normalization (originates from Day 3's normalize_broadcast)
# --------------------------------------------------------------------------
def zscore_normalize(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score normalize `arr` along `axis`, using broadcasting only (no loops).

    (value - mean) / std, computed per-slice along `axis`. For a 2D array
    with axis=0, this normalizes each column to mean 0, std 1 - the exact
    pattern from Day 3's normalize_broadcast.

    Args:
        arr: A 2D array.
        axis: The axis along which mean/std are computed (0 = per-column).

    Returns:
        A new array of the same shape, normalized.

    Raises:
        ValueError: If any slice along `axis` has zero standard deviation
            (which would cause division by zero).
    """
    mean = arr.mean(axis=axis)
    std = arr.std(axis=axis)

    if np.any(std == 0):
        raise ValueError(
            "Cannot normalize: at least one slice has zero standard "
            "deviation (all identical values), which would divide by zero."
        )

    return (arr - mean) / std


# --------------------------------------------------------------------------
# Ranking (built on Day 3's index-array ideas, using np.argsort)
# --------------------------------------------------------------------------
def rank_by_score(scores: np.ndarray, descending: bool = True) -> np.ndarray:
    """Rank entries by score using np.argsort, returning rank positions (not indices).

    Args:
        scores: A 1D array of scores, one per entry.
        descending: If True (default), the highest score gets rank 0.

    Returns:
        A 1D int array the same length as `scores`, where result[i] is
        the rank (0 = best) of entries[i]. This is the *rank of each
        original entry*, not a reordered list of indices - i.e. it
        answers "what place did student i finish in?" directly.
    """
    order = np.argsort(scores)
    if descending:
        order = order[::-1]

    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------
def write_text_report(lines: List[str], filepath: str) -> None:
    """Write a list of report lines to a plain text file, one per line."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def write_json_report(data: Dict[str, Any], filepath: str) -> None:
    """Write a dict to a JSON file, converting NumPy types to native Python first."""

    def _to_native(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(v) for v in obj]
        return obj

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_to_native(data), f, indent=2)
