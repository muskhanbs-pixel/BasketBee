"""
BasketBee - synthetic business data generator
=============================================

Creates realistic seed data for "BasketBee", a fictional Bengaluru grocery/FMCG
distributor with 3 warehouses, 15 dark stores and 60 products.

Run from the project root:
    python data/generate_data.py

Output (CSV files) goes to data/seed/:
    warehouses.csv, dark_stores.csv, suppliers.csv, products.csv,
    festivals.csv, sales_history.csv, inventory.csv, vehicles.csv

The data is deterministic (fixed random seed), so every teammate who runs
this script gets exactly the same numbers.

Demand patterns built in (so forecasting has something real to learn):
    - weekly cycle (weekends are busier)
    - salary-week bump (1st-5th of each month)
    - festival spikes (Diwali, Ganesh Chaturthi, Ugadi, Sankranti, Christmas)
      that hit some categories much harder than others
    - slow business growth over two years
    - random day-to-day noise
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START_DATE = "2024-10-01"
END_DATE = "2026-10-18"          # "today" in our story; history ends here
OUT_DIR = Path(__file__).resolve().parent / "seed"

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Warehouses (3) - approximate Bengaluru coordinates
# ---------------------------------------------------------------------------
WAREHOUSES = [
    # id, name, area, lat, lon, capacity_units, demand_share
    ("WH-SOUTH", "BasketBee South Hub", "Electronic City", 12.8452, 77.6602, 80000, 0.40),
    ("WH-NORTH", "BasketBee North Hub", "Yelahanka", 13.1007, 77.5963, 60000, 0.30),
    ("WH-EAST", "BasketBee East Hub", "Hoskote", 13.0707, 77.7982, 60000, 0.30),
]

# ---------------------------------------------------------------------------
# 2. Dark stores (15) - each served by one warehouse
# ---------------------------------------------------------------------------
DARK_STORES = [
    # id, area, lat, lon, served_by
    ("DS-01", "Koramangala", 12.9352, 77.6245, "WH-SOUTH"),
    ("DS-02", "HSR Layout", 12.9121, 77.6446, "WH-SOUTH"),
    ("DS-03", "BTM Layout", 12.9166, 77.6101, "WH-SOUTH"),
    ("DS-04", "Jayanagar", 12.9250, 77.5938, "WH-SOUTH"),
    ("DS-05", "JP Nagar", 12.9063, 77.5857, "WH-SOUTH"),
    ("DS-06", "Electronic City", 12.8399, 77.6770, "WH-SOUTH"),
    ("DS-07", "Hebbal", 13.0358, 77.5970, "WH-NORTH"),
    ("DS-08", "Malleshwaram", 13.0035, 77.5710, "WH-NORTH"),
    ("DS-09", "Rajajinagar", 12.9912, 77.5521, "WH-NORTH"),
    ("DS-10", "Yelahanka", 13.1007, 77.5963, "WH-NORTH"),
    ("DS-11", "Indiranagar", 12.9784, 77.6408, "WH-EAST"),
    ("DS-12", "Whitefield", 12.9698, 77.7500, "WH-EAST"),
    ("DS-13", "Marathahalli", 12.9569, 77.7011, "WH-EAST"),
    ("DS-14", "Bellandur", 12.9304, 77.6784, "WH-EAST"),
    ("DS-15", "KR Puram", 13.0076, 77.6950, "WH-EAST"),
]

# ---------------------------------------------------------------------------
# 3. Suppliers (8) - fictional names
# ---------------------------------------------------------------------------
SUPPLIERS = [
    # id, name, category focus, lead_time_days, on_time_rate, defect_rate, reliability note
    ("SUP-01", "Kaveri Fresh Dairy", "Dairy", 1, 0.96, 0.010),
    ("SUP-02", "Deccan Farm Produce", "Fruits & Vegetables", 1, 0.90, 0.030),
    ("SUP-03", "Malnad Staples Traders", "Staples", 4, 0.93, 0.008),
    ("SUP-04", "Sahyadri Snacks", "Snacks & Sweets", 3, 0.88, 0.012),
    ("SUP-05", "Coastal Beverages Co", "Beverages", 3, 0.94, 0.006),
    ("SUP-06", "Mysore Home Care", "Household", 5, 0.91, 0.005),
    ("SUP-07", "Nilgiri Personal Care", "Personal Care", 5, 0.95, 0.004),
    ("SUP-08", "Tungabhadra Agro Mills", "Staples", 6, 0.82, 0.015),  # cheaper but less reliable
]

# ---------------------------------------------------------------------------
# 4. Products (60)
#    sku, name, category, unit, unit_cost (INR), selling_price (INR),
#    shelf_life_days, base_daily_demand (units/day across all warehouses),
#    supplier_id
# ---------------------------------------------------------------------------
PRODUCTS = [
    # Dairy (perishable, high volume)
    ("DAI-001", "Toned Milk", "Dairy", "500 ml", 24, 28, 3, 900, "SUP-01"),
    ("DAI-002", "Full Cream Milk", "Dairy", "500 ml", 30, 34, 3, 450, "SUP-01"),
    ("DAI-003", "Curd", "Dairy", "400 g", 30, 35, 7, 380, "SUP-01"),
    ("DAI-004", "Paneer", "Dairy", "200 g", 75, 90, 10, 160, "SUP-01"),
    ("DAI-005", "Butter", "Dairy", "100 g", 48, 56, 120, 140, "SUP-01"),
    ("DAI-006", "Ghee", "Dairy", "500 ml", 290, 330, 270, 70, "SUP-01"),
    ("DAI-007", "Cheese Slices", "Dairy", "200 g", 120, 140, 90, 60, "SUP-01"),
    ("DAI-008", "Buttermilk", "Dairy", "200 ml", 12, 15, 5, 300, "SUP-01"),
    # Fruits & Vegetables (very perishable)
    ("FNV-001", "Onion", "Fruits & Vegetables", "1 kg", 30, 40, 20, 520, "SUP-02"),
    ("FNV-002", "Tomato", "Fruits & Vegetables", "1 kg", 25, 36, 6, 480, "SUP-02"),
    ("FNV-003", "Potato", "Fruits & Vegetables", "1 kg", 24, 32, 25, 450, "SUP-02"),
    ("FNV-004", "Banana (Robusta)", "Fruits & Vegetables", "1 dozen", 40, 55, 5, 300, "SUP-02"),
    ("FNV-005", "Apple (Shimla)", "Fruits & Vegetables", "1 kg", 140, 180, 20, 120, "SUP-02"),
    ("FNV-006", "Coriander Leaves", "Fruits & Vegetables", "100 g", 6, 10, 3, 350, "SUP-02"),
    ("FNV-007", "Green Chilli", "Fruits & Vegetables", "100 g", 6, 10, 7, 280, "SUP-02"),
    ("FNV-008", "Lemon", "Fruits & Vegetables", "250 g", 18, 25, 14, 200, "SUP-02"),
    ("FNV-009", "Carrot", "Fruits & Vegetables", "500 g", 22, 30, 10, 180, "SUP-02"),
    ("FNV-010", "Coconut", "Fruits & Vegetables", "1 pc", 28, 38, 30, 220, "SUP-02"),
    # Staples (long shelf life)
    ("STP-001", "Sona Masoori Rice", "Staples", "5 kg", 290, 340, 365, 160, "SUP-03"),
    ("STP-002", "Basmati Rice", "Staples", "1 kg", 110, 135, 365, 110, "SUP-03"),
    ("STP-003", "Whole Wheat Atta", "Staples", "5 kg", 210, 250, 180, 140, "SUP-03"),
    ("STP-004", "Toor Dal", "Staples", "1 kg", 140, 165, 270, 180, "SUP-08"),
    ("STP-005", "Moong Dal", "Staples", "1 kg", 120, 145, 270, 90, "SUP-08"),
    ("STP-006", "Sugar", "Staples", "1 kg", 42, 48, 365, 260, "SUP-03"),
    ("STP-007", "Iodised Salt", "Staples", "1 kg", 20, 25, 730, 150, "SUP-03"),
    ("STP-008", "Sunflower Oil", "Staples", "1 L", 130, 155, 270, 220, "SUP-08"),
    ("STP-009", "Groundnut Oil", "Staples", "1 L", 170, 200, 270, 80, "SUP-08"),
    ("STP-010", "Besan (Gram Flour)", "Staples", "500 g", 45, 55, 180, 90, "SUP-08"),
    ("STP-011", "Rava (Semolina)", "Staples", "500 g", 30, 38, 180, 110, "SUP-03"),
    ("STP-012", "Poha", "Staples", "500 g", 35, 42, 180, 100, "SUP-03"),
    # Snacks & Sweets (festival-sensitive)
    ("SNK-001", "Potato Chips", "Snacks & Sweets", "90 g", 32, 40, 120, 260, "SUP-04"),
    ("SNK-002", "Namkeen Mixture", "Snacks & Sweets", "400 g", 75, 95, 120, 140, "SUP-04"),
    ("SNK-003", "Cream Biscuits", "Snacks & Sweets", "150 g", 25, 30, 180, 300, "SUP-04"),
    ("SNK-004", "Dry Fruits Mix", "Snacks & Sweets", "250 g", 260, 320, 180, 45, "SUP-04"),
    ("SNK-005", "Cashew Nuts", "Snacks & Sweets", "250 g", 230, 285, 180, 50, "SUP-04"),
    ("SNK-006", "Gulab Jamun Tin", "Snacks & Sweets", "1 kg", 170, 210, 270, 25, "SUP-04"),
    ("SNK-007", "Milk Chocolate Bar", "Snacks & Sweets", "50 g", 40, 50, 270, 200, "SUP-04"),
    ("SNK-008", "Soan Papdi", "Snacks & Sweets", "500 g", 110, 140, 120, 20, "SUP-04"),
    ("SNK-009", "Instant Noodles", "Snacks & Sweets", "4-pack", 48, 56, 270, 280, "SUP-04"),
    ("SNK-010", "Rusk", "Snacks & Sweets", "300 g", 38, 45, 120, 130, "SUP-04"),
    # Beverages
    ("BEV-001", "Filter Coffee Powder", "Beverages", "500 g", 240, 290, 270, 90, "SUP-05"),
    ("BEV-002", "Tea Leaves", "Beverages", "500 g", 230, 275, 365, 120, "SUP-05"),
    ("BEV-003", "Packaged Drinking Water", "Beverages", "1 L", 15, 20, 180, 420, "SUP-05"),
    ("BEV-004", "Cola Soft Drink", "Beverages", "750 ml", 35, 40, 180, 200, "SUP-05"),
    ("BEV-005", "Mango Fruit Drink", "Beverages", "1 L", 75, 90, 180, 110, "SUP-05"),
    ("BEV-006", "Health Drink Powder", "Beverages", "500 g", 230, 270, 365, 60, "SUP-05"),
    ("BEV-007", "Tender Coconut Water", "Beverages", "200 ml", 30, 40, 60, 140, "SUP-05"),
    ("BEV-008", "Energy Drink", "Beverages", "250 ml", 95, 115, 365, 70, "SUP-05"),
    # Household
    ("HHC-001", "Detergent Powder", "Household", "1 kg", 110, 130, 730, 130, "SUP-06"),
    ("HHC-002", "Dishwash Liquid", "Household", "500 ml", 95, 115, 730, 120, "SUP-06"),
    ("HHC-003", "Floor Cleaner", "Household", "1 L", 160, 195, 730, 70, "SUP-06"),
    ("HHC-004", "Toilet Cleaner", "Household", "500 ml", 85, 100, 730, 65, "SUP-06"),
    ("HHC-005", "Garbage Bags", "Household", "30 pcs", 90, 110, 1095, 80, "SUP-06"),
    ("HHC-006", "Agarbatti", "Household", "100 sticks", 45, 60, 730, 90, "SUP-06"),
    ("HHC-007", "Diya Oil Lamps", "Household", "12 pcs", 60, 80, 1095, 10, "SUP-06"),
    # Personal Care
    ("PRC-001", "Bath Soap", "Personal Care", "4 x 100 g", 140, 170, 730, 150, "SUP-07"),
    ("PRC-002", "Shampoo", "Personal Care", "340 ml", 210, 250, 730, 70, "SUP-07"),
    ("PRC-003", "Toothpaste", "Personal Care", "150 g", 85, 100, 730, 140, "SUP-07"),
    ("PRC-004", "Coconut Hair Oil", "Personal Care", "250 ml", 95, 115, 730, 90, "SUP-07"),
    ("PRC-005", "Hand Wash", "Personal Care", "200 ml", 80, 99, 730, 75, "SUP-07"),
]

# ---------------------------------------------------------------------------
# 5. Festivals - approximate dates; edit festivals.csv if you want to refine.
#    window = days BEFORE the festival when shopping ramps up.
# ---------------------------------------------------------------------------
FESTIVALS = [
    # name, date, window_days
    ("Diwali", "2024-10-31", 7),
    ("Christmas", "2024-12-25", 4),
    ("Makar Sankranti", "2025-01-14", 4),
    ("Ugadi", "2025-03-30", 4),
    ("Ganesh Chaturthi", "2025-08-27", 5),
    ("Diwali", "2025-10-20", 7),
    ("Christmas", "2025-12-25", 4),
    ("Makar Sankranti", "2026-01-14", 4),
    ("Ugadi", "2026-03-19", 4),
    ("Ganesh Chaturthi", "2026-09-14", 5),
    ("Diwali", "2026-11-08", 7),      # upcoming - great forecasting demo!
    ("Christmas", "2026-12-25", 4),
]

# How strongly each festival lifts each category (multiplier at the peak)
FESTIVAL_LIFT = {
    "Diwali": {"Snacks & Sweets": 2.8, "Dairy": 1.5, "Staples": 1.4, "Household": 1.8,
               "Beverages": 1.3, "Personal Care": 1.2, "Fruits & Vegetables": 1.2},
    "Ganesh Chaturthi": {"Snacks & Sweets": 1.8, "Dairy": 1.4, "Staples": 1.3,
                         "Fruits & Vegetables": 1.4, "Household": 1.4},
    "Ugadi": {"Staples": 1.4, "Fruits & Vegetables": 1.5, "Snacks & Sweets": 1.4, "Dairy": 1.2},
    "Makar Sankranti": {"Staples": 1.4, "Snacks & Sweets": 1.6, "Fruits & Vegetables": 1.3,
                        "Dairy": 1.2},
    "Christmas": {"Snacks & Sweets": 1.6, "Beverages": 1.4, "Dairy": 1.2},
}

# A few products react extra strongly to festivals
PRODUCT_FESTIVAL_BOOST = {
    "HHC-007": 12.0,  # diya lamps at Diwali
    "SNK-008": 6.0,   # soan papdi
    "SNK-006": 4.0,   # gulab jamun
    "SNK-004": 3.0,   # dry fruits
    "SNK-005": 2.5,   # cashews
    "DAI-006": 1.6,   # ghee
    "HHC-006": 2.0,   # agarbatti
}

WEEKDAY_FACTOR = {0: 0.95, 1: 0.92, 2: 0.95, 3: 0.97, 4: 1.05, 5: 1.18, 6: 1.15}  # Mon..Sun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def festival_multiplier(dates: pd.DatetimeIndex, category: str, sku: str) -> np.ndarray:
    """Multiplier per date: ramps up over the festival window, peaks on the eve."""
    mult = np.ones(len(dates))
    for name, fdate, window in FESTIVALS:
        lift = FESTIVAL_LIFT.get(name, {}).get(category, 1.0)
        boost = PRODUCT_FESTIVAL_BOOST.get(sku, 1.0)
        if sku == "HHC-007" and name != "Diwali":   # diya lamps only spike at Diwali
            boost = 1.0
        if lift > 1.0:
            lift *= boost
        if lift <= 1.0:
            continue
        fday = pd.Timestamp(fdate)
        days_before = (fday - dates).days.values
        in_window = (days_before >= 0) & (days_before <= window)
        # linear ramp: smallest effect at start of window, full effect day before festival
        ramp = np.where(in_window, 1 - days_before / (window + 1), 0.0)
        mult *= 1 + (lift - 1) * ramp
        # demand dips a little just after the festival
        after = (days_before < 0) & (days_before >= -2)
        mult *= np.where(after, 0.85, 1.0)
    return mult


def daily_demand(base: float, dates: pd.DatetimeIndex, category: str, sku: str) -> np.ndarray:
    n = len(dates)
    weekday = np.array([WEEKDAY_FACTOR[d] for d in dates.dayofweek])
    salary = np.where(dates.day <= 5, 1.08, 1.0)
    growth = np.linspace(1.0, 1.25, n)                      # ~25% growth over 2 years
    summer = np.where(dates.month.isin([3, 4, 5]), 1.3, 1.0) if category == "Beverages" else 1.0
    fest = festival_multiplier(dates, category, sku)
    expected = base * weekday * salary * growth * summer * fest
    # negative-binomial-ish noise: Poisson around a gamma-perturbed mean
    noisy_mean = expected * rng.gamma(shape=25, scale=1 / 25, size=n)
    return rng.poisson(noisy_mean)


# ---------------------------------------------------------------------------
# Build tables
# ---------------------------------------------------------------------------
def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    warehouses = pd.DataFrame(WAREHOUSES, columns=[
        "warehouse_id", "name", "area", "lat", "lon", "capacity_units", "demand_share"])
    dark_stores = pd.DataFrame(DARK_STORES, columns=[
        "store_id", "area", "lat", "lon", "served_by_warehouse"])
    suppliers = pd.DataFrame(SUPPLIERS, columns=[
        "supplier_id", "name", "category", "lead_time_days", "on_time_rate", "defect_rate"])
    products = pd.DataFrame(PRODUCTS, columns=[
        "sku", "name", "category", "pack_size", "unit_cost_inr", "selling_price_inr",
        "shelf_life_days", "base_daily_demand", "supplier_id"])
    # inventory-policy fields used later by the Inventory agent
    products["holding_cost_pct_per_year"] = np.where(products["shelf_life_days"] <= 14, 0.40, 0.20)
    products["order_cost_inr"] = 500                       # fixed cost per purchase order
    products["case_pack"] = np.select(
        [products["base_daily_demand"] >= 300, products["base_daily_demand"] >= 100],
        [24, 12], default=6)
    festivals = pd.DataFrame(FESTIVALS, columns=["festival", "date", "window_days"])

    # --- sales history: one row per date x warehouse x sku ---
    rows = []
    for p in products.itertuples():
        for w in warehouses.itertuples():
            base = p.base_daily_demand * w.demand_share
            qty = daily_demand(base, dates, p.category, p.sku)
            rows.append(pd.DataFrame({
                "date": dates.date,
                "warehouse_id": w.warehouse_id,
                "sku": p.sku,
                "units_sold": qty,
            }))
    sales = pd.concat(rows, ignore_index=True)
    sales["revenue_inr"] = sales["units_sold"] * sales["sku"].map(
        products.set_index("sku")["selling_price_inr"])

    # --- current inventory snapshot (as of END_DATE) ---
    recent = (sales[pd.to_datetime(sales["date"]) > pd.Timestamp(END_DATE) - pd.Timedelta(days=14)]
              .groupby(["warehouse_id", "sku"])["units_sold"].mean().rename("avg_daily"))
    inv = recent.reset_index()
    inv = inv.merge(products[["sku", "shelf_life_days"]], on="sku")
    # days of cover: perishables are kept lean, long-life items deeper
    cover = np.where(inv["shelf_life_days"] <= 14, rng.uniform(1.5, 4, len(inv)),
                     rng.uniform(6, 25, len(inv)))
    inv["on_hand"] = (inv["avg_daily"] * cover).round().astype(int)

    # deliberately create stock-out risks for the demo (mostly Diwali items!)
    at_risk = [("WH-SOUTH", "SNK-004"), ("WH-SOUTH", "DAI-006"), ("WH-SOUTH", "STP-008"),
               ("WH-NORTH", "SNK-005"), ("WH-NORTH", "HHC-007"), ("WH-EAST", "STP-004"),
               ("WH-EAST", "SNK-002"), ("WH-EAST", "DAI-004")]
    for wh, sku in at_risk:
        m = (inv["warehouse_id"] == wh) & (inv["sku"] == sku)
        inv.loc[m, "on_hand"] = (inv.loc[m, "avg_daily"] * rng.uniform(0.6, 1.5)).round().astype(int)

    inv["reserved"] = (inv["on_hand"] * rng.uniform(0.0, 0.15, len(inv))).round().astype(int)
    in_transit_mask = rng.random(len(inv)) < 0.2
    inv["in_transit"] = np.where(in_transit_mask, (inv["avg_daily"] * rng.uniform(2, 6, len(inv))).round(), 0).astype(int)
    inv["updated_at"] = f"{END_DATE} 23:00:00"
    inv = inv[["warehouse_id", "sku", "on_hand", "reserved", "in_transit", "updated_at"]]

    # --- delivery fleet (for the TMS / routing agent later) ---
    vehicle_types = [("EV 3-wheeler", 350), ("Mini truck", 1200), ("Light truck", 3000)]
    vehicles = []
    vid = 1
    for w in warehouses.itertuples():
        for vtype, cap in vehicle_types:
            for _ in range(2):
                vehicles.append((f"VH-{vid:02d}", w.warehouse_id, vtype, cap,
                                 "available" if rng.random() > 0.15 else "maintenance"))
                vid += 1
    vehicles = pd.DataFrame(vehicles, columns=[
        "vehicle_id", "warehouse_id", "type", "capacity_kg", "status"])

    # --- save ---
    tables = {
        "warehouses": warehouses, "dark_stores": dark_stores, "suppliers": suppliers,
        "products": products, "festivals": festivals, "sales_history": sales,
        "inventory": inv, "vehicles": vehicles,
    }
    for name, df in tables.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    # --- summary ---
    print("BasketBee seed data generated in", OUT_DIR)
    for name, df in tables.items():
        print(f"  {name:<15} {len(df):>8,} rows")
    print(f"\nSales period: {START_DATE} to {END_DATE} ({len(dates)} days)")
    print(f"Total units sold: {sales['units_sold'].sum():,}")
    print(f"Total revenue:   INR {sales['revenue_inr'].sum():,.0f}")

    # quick sanity check: Diwali 2025 effect on Soan Papdi
    s = sales[sales["sku"] == "SNK-008"].groupby("date")["units_sold"].sum()
    s.index = pd.to_datetime(s.index)
    normal = s["2025-09-01":"2025-09-30"].mean()
    peak = s["2025-10-15":"2025-10-19"].mean()
    print(f"\nSanity check - Soan Papdi daily units: normal ~{normal:.0f}, "
          f"week before Diwali 2025 ~{peak:.0f}")


if __name__ == "__main__":
    build()