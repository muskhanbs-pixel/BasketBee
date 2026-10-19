"""
BasketBee - load the generated CSV data into the database.

Run from the project root (after `python data/generate_data.py`):
    python -m backend.db.seed

WARNING: this drops and recreates all tables, so it's a clean reset every time.
"""

import time
from pathlib import Path

import pandas as pd

from backend.core.config import PROJECT_ROOT
from backend.db import models  # noqa: F401  (import registers all tables on Base)
from backend.db.database import Base, engine

SEED_DIR = PROJECT_ROOT / "data" / "seed"

# Load order matters: parent tables before the tables that reference them
LOAD_ORDER = [
    ("warehouses", None),
    ("dark_stores", None),
    ("suppliers", None),
    ("products", None),
    ("festivals", ["date"]),
    ("vehicles", None),
    ("sales_history", ["date"]),
    ("inventory", ["updated_at"]),
]


def load_csv(name: str, date_cols: list[str] | None) -> pd.DataFrame:
    path = SEED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python data/generate_data.py` first.")
    df = pd.read_csv(path)
    for col in date_cols or []:
        df[col] = pd.to_datetime(df[col])
        if col == "date":
            df[col] = df[col].dt.date
    return df


def seed() -> None:
    start = time.perf_counter()
    print("Resetting tables...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    for name, date_cols in LOAD_ORDER:
        df = load_csv(name, date_cols)
        df.to_sql(name, engine, if_exists="append", index=False, chunksize=10_000)
        print(f"  loaded {name:<15} {len(df):>8,} rows")

    print(f"Done in {time.perf_counter() - start:.1f}s -> {engine.url}")


if __name__ == "__main__":
    seed()