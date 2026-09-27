"""common/viz_utils.py - Shared visualization utilities for Epic 3.

Mirrors `common/numpy_utils.py`'s pattern: re-imports the canonical
plotting functions from the day file where each was first built, rather
than duplicating any Matplotlib/Seaborn code a second time.

Where each function originates:
    - `CHARTS_DIR`, `plot_histogram`, `plot_boxplot_by_category`,
      `plot_scatter_correlation`, `plot_bar_chart`, `plot_line_chart`,
      `plot_dashboard_grid` -> Day 13 (Matplotlib)
    - `plot_correlation_heatmap`, `plot_pairplot`, `plot_violin_by_category`,
      `plot_regression_scatter`, `plot_countplot`,
      `plot_matplotlib_vs_seaborn_comparison` -> Day 14 (Seaborn)
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_epic3_dir = _project_root / "epic3_stats_viz"
if str(_epic3_dir) not in sys.path:
    sys.path.insert(0, str(_epic3_dir))

try:
    from epic3_stats_viz.day13_matplotlib_charts import (
        CHARTS_DIR,
        plot_bar_chart,
        plot_boxplot_by_category,
        plot_dashboard_grid,
        plot_histogram,
        plot_line_chart,
        plot_scatter_correlation,
    )
    from epic3_stats_viz.day14_seaborn_advanced import (
        plot_correlation_heatmap,
        plot_countplot,
        plot_matplotlib_vs_seaborn_comparison,
        plot_pairplot,
        plot_regression_scatter,
        plot_violin_by_category,
    )
except ImportError:
    from day13_matplotlib_charts import (
        CHARTS_DIR,
        plot_bar_chart,
        plot_boxplot_by_category,
        plot_dashboard_grid,
        plot_histogram,
        plot_line_chart,
        plot_scatter_correlation,
    )
    from day14_seaborn_advanced import (
        plot_correlation_heatmap,
        plot_countplot,
        plot_matplotlib_vs_seaborn_comparison,
        plot_pairplot,
        plot_regression_scatter,
        plot_violin_by_category,
    )

__all__ = [
    "CHARTS_DIR",
    "plot_histogram",
    "plot_boxplot_by_category",
    "plot_scatter_correlation",
    "plot_bar_chart",
    "plot_line_chart",
    "plot_dashboard_grid",
    "plot_correlation_heatmap",
    "plot_pairplot",
    "plot_violin_by_category",
    "plot_regression_scatter",
    "plot_countplot",
    "plot_matplotlib_vs_seaborn_comparison",
]
