"""common/stats_utils.py - Shared statistics utilities for Epic 3.

Mirrors the pattern `common/numpy_utils.py` established for Epic 1: this
module does NOT reimplement statistics logic - it re-imports the
canonical implementation from the day file where each function was
first built, so there is exactly one source of truth for each
computation. Day 15's `project_business_insights.py` imports everything
it needs from here (and from `viz_utils.py`) instead of reaching into
day11-day14 files directly.

Where each function originates:
    - `load_merged_dataset`      -> Day 11 (`_load_merged_dataset`)
    - `compute_central_tendency` -> Day 11
    - `compute_spread`           -> Day 11
    - `simple_probability`       -> Day 11
    - `conditional_probability`  -> Day 11
    - `bayes_theorem`            -> Day 11
    - `pearson_correlation`      -> Day 12
    - `correlation_matrix_report`-> Day 12
    - `strongest_and_weakest_pairs` -> Day 12
    - `fit_normal_distribution`  -> Day 12
    - `compute_zscores`          -> Day 12
    - `flag_zscore_outliers`     -> Day 12
    - `generate_distribution_samples` -> Day 12
    - `build_correlation_matrix` -> Day 14 (stats logic that happens to
      live in the Seaborn file, since it feeds the heatmap - moved here
      because it IS statistics, not plotting)
"""

from __future__ import annotations

import sys
from pathlib import Path

# common/ needs to reach into the sibling epic3_stats_viz/ package to
# import the canonical functions from Days 11-14 - make sure the project
# root is on sys.path regardless of how this module is imported.
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_epic3_dir = _project_root / "epic3_stats_viz"
if str(_epic3_dir) not in sys.path:
    sys.path.insert(0, str(_epic3_dir))

try:
    from epic3_stats_viz.day11_descriptive_stats import (
        _load_merged_dataset as load_merged_dataset,
        bayes_theorem,
        compute_central_tendency,
        compute_spread,
        conditional_probability,
        simple_probability,
    )
    from epic3_stats_viz.day12_correlation_distributions import (
        compute_zscores,
        correlation_matrix_report,
        fit_normal_distribution,
        flag_zscore_outliers,
        generate_distribution_samples,
        pearson_correlation,
        strongest_and_weakest_pairs,
    )
    from epic3_stats_viz.day14_seaborn_advanced import build_correlation_matrix
except ImportError:
    from day11_descriptive_stats import (
        _load_merged_dataset as load_merged_dataset,
        bayes_theorem,
        compute_central_tendency,
        compute_spread,
        conditional_probability,
        simple_probability,
    )
    from day12_correlation_distributions import (
        compute_zscores,
        correlation_matrix_report,
        fit_normal_distribution,
        flag_zscore_outliers,
        generate_distribution_samples,
        pearson_correlation,
        strongest_and_weakest_pairs,
    )
    from day14_seaborn_advanced import build_correlation_matrix

__all__ = [
    "load_merged_dataset",
    "compute_central_tendency",
    "compute_spread",
    "simple_probability",
    "conditional_probability",
    "bayes_theorem",
    "pearson_correlation",
    "correlation_matrix_report",
    "strongest_and_weakest_pairs",
    "fit_normal_distribution",
    "compute_zscores",
    "flag_zscore_outliers",
    "generate_distribution_samples",
    "build_correlation_matrix",
]
