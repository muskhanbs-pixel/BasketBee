"""
BasketBee - forecast evaluation (evidence for the report & Review II).

Question: does our model really beat a simple baseline, especially around festivals?

Method (a "backtest"):
  * Pretend today is 2025-10-05 - two weeks before Diwali 2025.
  * Train on all sales BEFORE that date only.
  * Forecast the next 28 days (covers the whole Diwali 2025 rush).
  * Compare with what actually sold.
  * Repeat for all 60 products x 3 warehouses = 180 forecasts.

Run from the project root (no server needed, takes 1-2 minutes):
    python -m eval.evaluate_forecast

Output: a summary in the terminal + eval/results/forecast_backtest.csv
"""

import time

import numpy as np
import pandas as pd

from backend.tools.adapters import ERPAdapter
from backend.tools.forecasting import BASELINE_WINDOW, _fit_predict

CUTOFF = pd.Timestamp("2025-10-06")
HORIZON = 28
WAREHOUSES = ["WH-SOUTH", "WH-NORTH", "WH-EAST"]


def main():
    erp = ERPAdapter.in_process()
    festivals = erp.festivals()
    products = pd.DataFrame(erp._get("/erp/products"))

    rows = []
    start = time.perf_counter()
    for i, p in enumerate(products.itertuples(), 1):
        for wh in WAREHOUSES:
            hist = erp.sales_history(p.sku, wh)
            train = hist[hist["date"] < CUTOFF]
            test = hist[hist["date"] >= CUTOFF].head(HORIZON)
            actual = test["units_sold"].values
            pred = _fit_predict(train, pd.DatetimeIndex(test["date"]), festivals)
            base = np.full(len(test), train["units_sold"].tail(BASELINE_WINDOW).mean())
            rows.append({
                "sku": p.sku, "name": p.name, "category": p.category, "warehouse_id": wh,
                "actual_units": actual.sum(),
                "model_abs_error": np.abs(actual - pred).sum(),
                "baseline_abs_error": np.abs(actual - base).sum(),
                "actual_peak": actual.max(), "model_peak": round(pred.max(), 1),
            })
        print(f"\r  {i}/{len(products)} products done", end="")
    print(f"  ({time.perf_counter() - start:.0f}s)\n")

    df = pd.DataFrame(rows)
    df["model_wape_pct"] = (100 * df["model_abs_error"] / df["actual_units"]).round(1)
    df["baseline_wape_pct"] = (100 * df["baseline_abs_error"] / df["actual_units"]).round(1)
    df["model_wins"] = df["model_wape_pct"] < df["baseline_wape_pct"]

    out = pd.io.common.Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "forecast_backtest.csv", index=False)

    def overall(g):
        return pd.Series({
            "model_wape_%": round(100 * g["model_abs_error"].sum() / g["actual_units"].sum(), 1),
            "baseline_wape_%": round(100 * g["baseline_abs_error"].sum() / g["actual_units"].sum(), 1),
            "model_wins": f"{g['model_wins'].sum()}/{len(g)}",
        })

    print("Backtest: Diwali 2025 window (28 days from 2025-10-06)\n")
    print(df.groupby("category").apply(overall, include_groups=False).to_string())
    total = overall(df)
    print(f"\nOVERALL  model {total['model_wape_%']}%  vs  baseline {total['baseline_wape_%']}%  "
          f"(model better on {total['model_wins']} forecasts)")
    print(f"Improvement: {100 * (1 - total['model_wape_%'] / total['baseline_wape_%']):.0f}% less error")
    print(f"\nDetails saved to {out / 'forecast_backtest.csv'}")


if __name__ == "__main__":
    main()