"""Day 6: Series, DataFrames & Data Ingestion

This module covers pandas Series and DataFrame creation, multi-format
data ingestion (CSV/Excel/JSON), and the core DataFrame inspection
methods used throughout the rest of Epic 2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CANONICAL_CSV = DATA_DIR / "ecommerce_orders.csv"
CANONICAL_XLSX = DATA_DIR / "ecommerce_orders.xlsx"
CANONICAL_JSON = DATA_DIR / "ecommerce_orders.json"


# --------------------------------------------------------------------------
# 1. Series creation
# --------------------------------------------------------------------------
def series_demo() -> None:
    """Demonstrate creating a Series from a list, dict, and NumPy array,
    including custom index labels."""
    from_list = pd.Series([10, 20, 30, 40])
    logger.info("Series from list (default integer index):\n%s", from_list)

    from_dict = pd.Series({"a": 100, "b": 200, "c": 300})
    logger.info("Series from dict (keys become the index):\n%s", from_dict)

    from_array = pd.Series(np.array([1.5, 2.5, 3.5]))
    logger.info("Series from NumPy array:\n%s", from_array)

    custom_index = pd.Series([5, 6, 7], index=["mon", "tue", "wed"])
    logger.info("Series with custom index labels:\n%s", custom_index)
    logger.info("Accessing by custom label custom_index['tue']: %s", custom_index["tue"])


# --------------------------------------------------------------------------
# 2. DataFrame creation
# --------------------------------------------------------------------------
def dataframe_creation_demo() -> None:
    """Demonstrate creating a DataFrame from a dict of lists, list of dicts,
    and a NumPy 2D array."""
    from_dict_of_lists = pd.DataFrame({
        "name": ["Alice", "Bob", "Carol"],
        "age": [25, 30, 35],
    })
    logger.info("DataFrame from dict of lists:\n%s", from_dict_of_lists)

    from_list_of_dicts = pd.DataFrame([
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
        {"name": "Carol", "age": 35},
    ])
    logger.info("DataFrame from list of dicts:\n%s", from_list_of_dicts)

    from_array = pd.DataFrame(
        np.arange(12).reshape(3, 4),
        columns=["w", "x", "y", "z"],
    )
    logger.info("DataFrame from NumPy 2D array:\n%s", from_array)


# --------------------------------------------------------------------------
# 3. Multi-format data ingestion
# --------------------------------------------------------------------------
def load_dataset(filepath: str) -> pd.DataFrame:
    """Load a dataset from `filepath`, auto-detecting the format by extension.

    Supports .csv, .xlsx (and .xls), and .json.

    The 'order_date' column (when present) is parsed into a genuine
    datetime64 dtype rather than left as a plain string - this is what
    makes it a real THIRD mixed type (numeric, string, date) rather than
    just a string that happens to look like a date. CSV, Excel, and JSON
    all store dates as plain text on disk, so every format needs this
    same explicit parsing step after loading, not just CSV.

    Args:
        filepath: Path to the data file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If `filepath` does not exist.
        ValueError: If the file extension is not one of the supported formats.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}' for {filepath}. "
            f"Supported formats: .csv, .xlsx, .xls, .json"
        )

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"])

    return df


# --------------------------------------------------------------------------
# 4. DataFrame inspection methods
# --------------------------------------------------------------------------
def inspection_demo(df: pd.DataFrame) -> None:
    """Demonstrate the core DataFrame inspection methods."""
    logger.info(".head():\n%s", df.head())
    logger.info(".tail():\n%s", df.tail())
    logger.info(".shape: %s", df.shape)
    logger.info(".columns: %s", list(df.columns))
    logger.info(".dtypes:\n%s", df.dtypes)

    # .info() prints directly rather than returning a string, so it's
    # captured via a buffer to route it through logging like everything else.
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    logger.info(".info():\n%s", buf.getvalue())

    logger.info(".describe():\n%s", df.describe())


# --------------------------------------------------------------------------
# 5. Practical exercise: dataframe_summary
# --------------------------------------------------------------------------
def dataframe_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize a DataFrame's shape, dtypes, null counts, and memory usage.

    Args:
        df: The DataFrame to summarize.

    Returns:
        A dict with keys: 'shape', 'dtypes', 'null_counts',
        'total_nulls', 'memory_usage_bytes'.
    """
    return {
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "total_nulls": int(df.isnull().sum().sum()),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }


# --------------------------------------------------------------------------
# 6. Practical exercise: select_columns_rows
# --------------------------------------------------------------------------
def select_columns_rows(df: pd.DataFrame) -> None:
    """Demonstrate .loc, .iloc, column selection, and row slicing on `df`.

    .loc selects by LABEL (row/column names or a boolean mask).
    .iloc selects by numeric POSITION, exactly like NumPy/list indexing -
    this is the same distinction Day 3 covered for NumPy arrays, now
    applied to pandas' label-aware indexing.
    """
    logger.info("Single column selection df['order_id']:\n%s", df["order_id"].head())
    logger.info("Multiple column selection df[['order_id', 'status']]:\n%s",
                df[["order_id", "status"]].head())

    logger.info(".iloc[0] (first row, by position):\n%s", df.iloc[0])
    logger.info(".iloc[0:3] (first 3 rows, by position):\n%s", df.iloc[0:3])
    logger.info(".iloc[0:3, 1:3] (rows 0-2, columns 1-2, by position):\n%s", df.iloc[0:3, 1:3])

    logger.info(".loc[0] (row labeled 0):\n%s", df.loc[0])
    logger.info(
        ".loc[0:2, ['order_id', 'status']] (rows labeled 0-2 inclusive, named columns):\n%s",
        df.loc[0:2, ["order_id", "status"]],
    )

    # Boolean-mask based row selection via .loc (same masking idea as Day 3)
    high_value = df.loc[df["unit_price"] > 400]
    logger.info(".loc with boolean mask (unit_price > 400): %d matching rows", len(high_value))


# --------------------------------------------------------------------------
# 7. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 6: Series, DataFrames & Data Ingestion ===")

    logger.info("--- Series creation demo ---")
    series_demo()

    logger.info("--- DataFrame creation demo ---")
    dataframe_creation_demo()

    logger.info("--- Loading canonical dataset (CSV) ---")
    df = load_dataset(str(CANONICAL_CSV))
    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])

    logger.info("--- Verifying all 3 formats load correctly ---")
    df_csv = load_dataset(str(CANONICAL_CSV))
    df_xlsx = load_dataset(str(CANONICAL_XLSX))
    df_json = load_dataset(str(CANONICAL_JSON))
    logger.info("CSV shape: %s | Excel shape: %s | JSON shape: %s",
                df_csv.shape, df_xlsx.shape, df_json.shape)

    logger.info("--- Inspection methods demo ---")
    inspection_demo(df)

    logger.info("--- dataframe_summary() report ---")
    summary = dataframe_summary(df)
    logger.info("shape: %s", summary["shape"])
    logger.info("dtypes: %s", summary["dtypes"])
    logger.info("null_counts: %s", summary["null_counts"])
    logger.info("total_nulls: %s", summary["total_nulls"])
    logger.info("memory_usage_bytes: %s", summary["memory_usage_bytes"])

    logger.info("--- select_columns_rows demo ---")
    select_columns_rows(df)


if __name__ == "__main__":
    main()
