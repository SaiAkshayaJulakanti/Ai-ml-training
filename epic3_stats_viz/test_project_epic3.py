"""Pytest suite for project_business_insights.py (Epic 3 capstone).

Run with: pytest -v
Coverage: pytest --cov=epic3_stats_viz --cov=common
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pypdf

file_dir = Path(__file__).parent
project_root = file_dir.parent
for p in (file_dir, project_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from epic3_stats_viz.project_business_insights import (
        build_insights_and_recommendations,
        compile_pdf_report,
        generate_full_report,
        write_insights_markdown,
        _compute_stats_summary,
    )
except ImportError:
    from project_business_insights import (
        build_insights_and_recommendations,
        compile_pdf_report,
        generate_full_report,
        write_insights_markdown,
        _compute_stats_summary,
    )

# Imported via the SAME dotted path (`epic3_stats_viz.dayNN...`) that
# common/stats_utils.py and common/viz_utils.py use internally - Python
# caches modules by their exact import path, so importing "day11_..."
# bare here would load a SECOND, distinct copy of the module (with
# different function objects), breaking the identity checks below even
# though the source code is identical.
try:
    from epic3_stats_viz import (
        day11_descriptive_stats,
        day12_correlation_distributions,
        day13_matplotlib_charts,
        day14_seaborn_advanced,
    )
except ImportError:
    import day11_descriptive_stats
    import day12_correlation_distributions
    import day13_matplotlib_charts
    import day14_seaborn_advanced
from common import stats_utils, viz_utils


def _assert_valid_pdf(path: Path) -> None:
    assert path.exists(), f"expected a file at {path}"
    assert path.stat().st_size > 0, f"{path} was created but is empty"
    with open(path, "rb") as f:
        assert f.read(5) == b"%PDF-"


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """A small but realistic stand-in for the merged Epic 2 dataset, with
    all the columns generate_full_report/compile_pdf_report expect."""
    rng = np.random.default_rng(21)
    n = 60
    categories = rng.choice(["Electronics", "Clothing", "Books"], size=n)
    return pd.DataFrame({
        "order_id": range(1, n + 1),
        "quantity": rng.integers(1, 10, size=n),
        "unit_price": rng.normal(200, 60, size=n).clip(5, None),
        "standard_discount_pct": rng.uniform(0, 15, size=n),
        "product_category": categories,
        "order_date": pd.date_range("2026-01-01", periods=n, freq="3D"),
    })


# --------------------------------------------------------------------------
# End-to-end orchestration
# --------------------------------------------------------------------------
def test_generate_full_report_runs_without_exceptions(tmp_path, sample_df):
    output_pdf = tmp_path / "report.pdf"
    generate_full_report(sample_df, str(output_pdf))  # must not raise
    _assert_valid_pdf(output_pdf)


def test_generate_full_report_writes_insights_markdown_alongside_pdf(tmp_path, sample_df):
    output_pdf = tmp_path / "report.pdf"
    generate_full_report(sample_df, str(output_pdf))
    insights_path = tmp_path / "INSIGHTS.md"
    assert insights_path.exists()
    assert insights_path.stat().st_size > 0


def test_generate_full_report_pdf_has_multiple_pages(tmp_path, sample_df):
    output_pdf = tmp_path / "report.pdf"
    generate_full_report(sample_df, str(output_pdf))
    reader = pypdf.PdfReader(str(output_pdf))
    # title + stats table + 11 charts + findings + recommendations = 15
    assert len(reader.pages) >= 10


# --------------------------------------------------------------------------
# Insights content
# --------------------------------------------------------------------------
def test_build_insights_and_recommendations_counts(sample_df):
    stats_summary = _compute_stats_summary(sample_df)
    findings, recommendations = build_insights_and_recommendations(stats_summary)
    assert 5 <= len(findings) <= 7
    assert len(recommendations) == 3


def test_insights_markdown_contains_required_sections(tmp_path, sample_df):
    stats_summary = _compute_stats_summary(sample_df)
    findings, recommendations = build_insights_and_recommendations(stats_summary)
    out = tmp_path / "INSIGHTS.md"
    write_insights_markdown(findings, recommendations, str(out))

    content = out.read_text(encoding="utf-8")
    assert "## Key Findings" in content
    assert "## Recommendations" in content
    for finding in findings:
        assert finding in content


def test_findings_reflect_real_data_not_placeholders(sample_df):
    # The mean unit_price computed independently should show up verbatim
    # (to 2 decimal places) inside one of the generated findings - proof
    # the findings are pulled from real numbers, not hardcoded text.
    stats_summary = _compute_stats_summary(sample_df)
    findings, _ = build_insights_and_recommendations(stats_summary)
    expected_mean = f"{stats_summary['central_tendency']['unit_price']['mean']:.2f}"
    assert any(expected_mean in f for f in findings)


# --------------------------------------------------------------------------
# compile_pdf_report as a standalone unit (independent of the full pipeline)
# --------------------------------------------------------------------------
def test_compile_pdf_report_standalone(tmp_path, sample_df):
    # Two throwaway chart images, built directly with matplotlib rather
    # than going through the whole chart-generation pipeline.
    import matplotlib.pyplot as plt

    chart1 = tmp_path / "chart1.png"
    chart2 = tmp_path / "chart2.png"
    for path in (chart1, chart2):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [3, 1, 2])
        fig.savefig(path)
        plt.close(fig)

    stats_summary = _compute_stats_summary(sample_df)
    output_pdf = tmp_path / "standalone_report.pdf"
    compile_pdf_report([str(chart1), str(chart2)], stats_summary, str(output_pdf))

    _assert_valid_pdf(output_pdf)
    reader = pypdf.PdfReader(str(output_pdf))
    # title + stats table + 2 charts + findings + recommendations = 6
    assert len(reader.pages) == 6


# --------------------------------------------------------------------------
# Refactored common/ utils produce identical output to the Day 11-14 originals
# --------------------------------------------------------------------------
def test_stats_utils_are_the_same_objects_as_day11_day12_originals():
    # Identity (not just equal output) proves these are re-exports of the
    # single canonical implementation, not separately maintained copies.
    assert stats_utils.compute_central_tendency is day11_descriptive_stats.compute_central_tendency
    assert stats_utils.compute_spread is day11_descriptive_stats.compute_spread
    assert stats_utils.pearson_correlation is day12_correlation_distributions.pearson_correlation
    assert stats_utils.build_correlation_matrix is day14_seaborn_advanced.build_correlation_matrix


def test_viz_utils_are_the_same_objects_as_day13_day14_originals():
    assert viz_utils.plot_histogram is day13_matplotlib_charts.plot_histogram
    assert viz_utils.plot_dashboard_grid is day13_matplotlib_charts.plot_dashboard_grid
    assert viz_utils.plot_correlation_heatmap is day14_seaborn_advanced.plot_correlation_heatmap


def test_stats_utils_output_matches_original_on_real_data(sample_df):
    # Belt-and-suspenders: even setting identity aside, run both "paths"
    # on the same data and confirm the numbers agree.
    data = sample_df["unit_price"].to_numpy(dtype=float)
    original = day11_descriptive_stats.compute_central_tendency(data)
    via_common = stats_utils.compute_central_tendency(data)
    assert original == via_common
