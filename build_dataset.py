"""
============================================================
  BUILD_DATASET.PY  –  Smart Retail Master Dataset Builder
  Merges sales_data.csv + retail_store_inventory.csv and
  engineers all features needed by the ML pipeline.

  FIXES vs original:
  - Category map now covers both directions (Food↔Groceries)
  - inv_agg month grouper uses named column after reset to
    avoid KeyError on merge
  - Gross_Profit uses per-unit cost correctly
  - Inventory_Coverage guards against div-by-zero properly
  - All output paths use the project folder constant
============================================================
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smart_retail_ml")
os.makedirs(PROJECT_DIR, exist_ok=True)

# ── 1. Load raw data ──────────────────────────────────────
SALES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sales_data.csv")
INV_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retail_store_inventory.csv")

sales = pd.read_csv(SALES_PATH, parse_dates=["Sale_Date"])
inv   = pd.read_csv(INV_PATH,   parse_dates=["Date"])

# Normalise category names so they match across both files
# sales_data uses: Clothing, Electronics, Food, Furniture
# inventory  uses: Clothing, Electronics, Groceries, Furniture  → map Groceries→Food
cat_map = {"Groceries": "Food"}
inv["Category"] = inv["Category"].replace(cat_map)

print(f"Sales shape      : {sales.shape}")
print(f"Inventory shape  : {inv.shape}")
print(f"Sales categories : {sorted(sales['Product_Category'].unique())}")
print(f"Inv  categories  : {sorted(inv['Category'].unique())}")
print(f"Sales regions    : {sorted(sales['Region'].unique())}")
print(f"Inv  regions     : {sorted(inv['Region'].unique())}")

# ── 2. Aggregate inventory to (Month, Region, Category) ──
inv["Month"] = inv["Date"].dt.to_period("M").dt.to_timestamp()

inv_agg = (
    inv.groupby(["Month", "Region", "Category"])
    .agg(
        Inventory_Level    = ("Inventory Level",    "mean"),
        Units_Sold_Online  = ("Units Sold",         "sum"),
        Units_Ordered      = ("Units Ordered",      "sum"),
        Demand_Forecast    = ("Demand Forecast",    "mean"),
        Avg_Price          = ("Price",              "mean"),
        Avg_Discount_Inv   = ("Discount",           "mean"),
        Weather_Mode       = ("Weather Condition",  lambda x: x.mode().iloc[0]),
        Holiday_Days       = ("Holiday/Promotion",  "sum"),
        Competitor_Pricing = ("Competitor Pricing", "mean"),
        Seasonality        = ("Seasonality",        lambda x: x.mode().iloc[0]),
    )
    .reset_index()
)

# ── 3. Prepare & aggregate sales data ────────────────────
sales["Month"] = sales["Sale_Date"].dt.to_period("M").dt.to_timestamp()

sales_agg = (
    sales.groupby(["Month", "Region", "Product_Category"])
    .agg(
        Total_Sales_Amount  = ("Sales_Amount",   "sum"),
        Total_Qty_Sold      = ("Quantity_Sold",  "sum"),
        Avg_Unit_Cost       = ("Unit_Cost",      "mean"),   # avg cost per transaction
        Avg_Unit_Price      = ("Unit_Price",     "mean"),   # avg price per transaction
        Avg_Discount_Sales  = ("Discount",       "mean"),
        Num_Transactions    = ("Product_ID",     "count"),
        New_Customers       = ("Customer_Type",  lambda x: (x == "New").sum()),
        Returning_Customers = ("Customer_Type",  lambda x: (x == "Returning").sum()),
        Online_Txn          = ("Sales_Channel",  lambda x: (x == "Online").sum()),
        Retail_Txn          = ("Sales_Channel",  lambda x: (x == "Retail").sum()),
    )
    .reset_index()
    .rename(columns={"Product_Category": "Category"})
)

# ── 4. Merge ──────────────────────────────────────────────
df = sales_agg.merge(inv_agg, on=["Month", "Region", "Category"], how="left")
print(f"\nMerged shape     : {df.shape}")
print(f"Merge null check (Inventory_Level): {df['Inventory_Level'].isna().sum()} nulls")

# ── 5. Feature Engineering ────────────────────────────────

# --- Core financials ---
# Gross profit = revenue − cost of goods sold
# Unit_Cost in sales_data is the cost for that transaction row
# so COGS per month = Avg_Unit_Cost × Num_Transactions
df["COGS"]               = df["Avg_Unit_Cost"] * df["Num_Transactions"]
df["Gross_Profit"]       = df["Total_Sales_Amount"] - df["COGS"]
df["Profit_Margin_Pct"]  = (df["Gross_Profit"] / df["Total_Sales_Amount"].replace(0, np.nan)) * 100
df["Avg_Revenue_Per_Txn"]= df["Total_Sales_Amount"] / df["Num_Transactions"].replace(0, np.nan)

# --- Transportation cost estimate (2% of COGS) ---
df["Transport_Cost_Est"] = df["COGS"] * 0.02

# --- Net Profit after transport ---
df["Net_Profit"]         = df["Gross_Profit"] - df["Transport_Cost_Est"]
df["Net_Profit_Margin"]  = (df["Net_Profit"] / df["Total_Sales_Amount"].replace(0, np.nan)) * 100

# --- Channel split ratios ---
df["Spot_Sales_Ratio"]   = df["Retail_Txn"]  / df["Num_Transactions"].replace(0, np.nan)
df["Online_Sales_Ratio"] = df["Online_Txn"]   / df["Num_Transactions"].replace(0, np.nan)

# --- Price competitiveness ---
df["Price_vs_Competitor"]= df["Avg_Unit_Price"] - df["Competitor_Pricing"].fillna(df["Avg_Unit_Price"])

# --- Demand fulfilment rate (capped 0–1) ---
df["Fulfillment_Rate"]   = (
    df["Units_Sold_Online"] / df["Demand_Forecast"].replace(0, np.nan)
).clip(0, 1)

# --- Inventory adequacy in days (inventory / daily sales rate) ---
daily_sales = (df["Total_Qty_Sold"] / 30).replace(0, np.nan)
df["Inventory_Coverage"] = df["Inventory_Level"] / daily_sales

# --- Time features ---
df["Year"]     = df["Month"].dt.year
df["MonthNum"] = df["Month"].dt.month
df["Quarter"]  = df["Month"].dt.quarter

# --- Recommended selling price (target 40% gross margin) ---
df["Recommended_Price"]      = df["Avg_Unit_Cost"] * (1 / (1 - 0.40))   # = cost / 0.60

# --- Qty to order: forecast + 15% safety stock ---
df["Recommended_Order_Qty"]  = (
    df["Demand_Forecast"].fillna(df["Units_Ordered"]) * 1.15
).round()

# --- Month-over-Month growth (per region + category) ---
df = df.sort_values(["Region", "Category", "Month"]).reset_index(drop=True)
df["Sales_Growth_MoM"]  = (
    df.groupby(["Region", "Category"])["Total_Sales_Amount"]
    .pct_change() * 100
)
df["Profit_Growth_MoM"] = (
    df.groupby(["Region", "Category"])["Net_Profit"]
    .pct_change() * 100
)

# ── 6. Sanity check ──────────────────────────────────────
print(f"\nNet_Profit_Margin stats:\n{df['Net_Profit_Margin'].describe().round(2)}")
print(f"Nulls per column:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# ── 7. Save master dataset ───────────────────────────────
out = os.path.join(PROJECT_DIR, "master_dataset.csv")
df.to_csv(out, index=False)
print(f"\n✅  Master dataset saved → {out}")
print(f"   Shape : {df.shape}")
print(f"   Columns ({len(df.columns)}):")
for c in df.columns:
    print(f"     {c}")