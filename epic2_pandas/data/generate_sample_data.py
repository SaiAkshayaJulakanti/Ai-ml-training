"""epic2_pandas/data/generate_sample_data.py

Generates the canonical e-commerce orders dataset used throughout
Days 6-9. Run this once to (re)create the sample CSV/Excel/JSON files
under epic2_pandas/data/.

This is a data-generation utility, not part of the day6 exercise
functions themselves - it exists so the dataset's origin is documented
and reproducible rather than being a mystery file that just "appears".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUM_ORDERS = 150

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn", "Drew"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
CATEGORIES = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Toys"]
REGIONS = ["North", "South", "East", "West", "Central"]
STATUSES = ["Delivered", "Pending", "Cancelled", "Shipped"]
STATUS_WEIGHTS = [0.55, 0.15, 0.10, 0.20]
NUM_MISSING_NAMES = 5  # intentionally introduced nulls, for null-count demos


def generate_orders_dataframe(num_orders: int = NUM_ORDERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate the canonical e-commerce orders dataset.

    8 columns with mixed types: numeric (order_id, quantity, unit_price),
    string/categorical (customer_name, product_category, region, status),
    and date (order_date) - satisfying the "5+ columns, mixed types"
    requirement with room to spare.

    A handful of customer_name values are intentionally set to null, so
    downstream null-count demonstrations (dataframe_summary) have
    something real to report.
    """
    rng = np.random.default_rng(seed)

    customer_names = [
        f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(num_orders)
    ]
    order_dates = pd.date_range("2026-01-01", "2026-08-31", periods=num_orders).strftime("%Y-%m-%d")

    df = pd.DataFrame({
        "order_id": np.arange(1001, 1001 + num_orders),
        "customer_name": customer_names,
        "product_category": rng.choice(CATEGORIES, num_orders),
        "quantity": rng.integers(1, 10, num_orders),
        "unit_price": np.round(rng.uniform(5.0, 500.0, num_orders), 2),
        "order_date": order_dates,
        "region": rng.choice(REGIONS, num_orders),
        "status": rng.choice(STATUSES, num_orders, p=STATUS_WEIGHTS),
    })

    missing_idx = rng.choice(num_orders, size=NUM_MISSING_NAMES, replace=False)
    df.loc[missing_idx, "customer_name"] = None

    return df


def save_all_formats(df: pd.DataFrame, output_dir: str) -> None:
    """Save `df` to CSV, Excel (.xlsx), and JSON in `output_dir`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "ecommerce_orders.csv", index=False)
    df.to_excel(out / "ecommerce_orders.xlsx", index=False, engine="openpyxl")
    df.to_json(out / "ecommerce_orders.json", orient="records", date_format="iso")


if __name__ == "__main__":
    dataset = generate_orders_dataframe()
    save_all_formats(dataset, str(Path(__file__).parent))
    print(f"Generated {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print(dataset.head())
