"""
============================================================
  ML_PIPELINE.PY  –  Smart Retail Intelligence Engine

  Models trained:
    A. Sales Amount Predictor         (RandomForest Regressor)
    B. Net Profit Margin Predictor    (GradientBoosting Regressor)
    C. Sell Quantity Predictor        (RandomForest Regressor)
    D. High / Low Profit Classifier   (RandomForest Classifier)

  Outputs  (all saved to ./smart_retail_ml/):
    master_dataset.csv       ← must already exist (run build_dataset.py first)
    predictions.csv
    monthly_summary.csv
    yearly_summary.csv
    improvement_report.csv
    feature_importance.csv
    model_scores.csv
    future_predictions.csv   ← generates 12-month forward forecast per segment
============================================================
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor,
                               RandomForestClassifier)
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, r2_score,
                              classification_report, accuracy_score)

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smart_retail_ml")
os.makedirs(PROJECT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
# 0. Load master dataset
# ══════════════════════════════════════════════════════════
master_path = os.path.join(PROJECT_DIR, "master_dataset.csv")
if not os.path.exists(master_path):
    raise FileNotFoundError(
        f"Master dataset not found at {master_path}.\n"
        "Please run build_dataset.py first."
    )

df = pd.read_csv(master_path, parse_dates=["Month"])
print(f"Loaded master dataset: {df.shape}")

# ══════════════════════════════════════════════════════════
# 1. Encode categoricals
# ══════════════════════════════════════════════════════════
label_encoders = {}
for col in ["Region", "Category", "Weather_Mode", "Seasonality"]:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ══════════════════════════════════════════════════════════
# 2. Feature set  (no target-leaking columns)
# ══════════════════════════════════════════════════════════
FEATURE_COLS = [
    "Region_enc", "Category_enc",
    "Year", "MonthNum", "Quarter",
    "Avg_Unit_Cost",
    "Avg_Discount_Sales",
    "Num_Transactions",
    "New_Customers", "Returning_Customers",
    "Online_Txn", "Retail_Txn",
    "Inventory_Level", "Units_Ordered", "Demand_Forecast",
    "Avg_Discount_Inv", "Holiday_Days", "Competitor_Pricing",
    "Seasonality_enc", "Weather_Mode_enc",
    "Price_vs_Competitor",
    "Fulfillment_Rate",
    "Recommended_Order_Qty",
]

TARGET_COLS = ["Total_Sales_Amount", "Net_Profit_Margin", "Total_Qty_Sold"]
df_model = df.dropna(subset=FEATURE_COLS + TARGET_COLS).copy().reset_index(drop=True)

X = df_model[FEATURE_COLS].apply(lambda c: c.fillna(c.median()))

print(f"Modelling rows   : {len(df_model)}")
print(f"Feature columns  : {len(FEATURE_COLS)}")

scores_log = {}

# ══════════════════════════════════════════════════════════
# 3. MODEL A – Sales Amount Predictor
# ══════════════════════════════════════════════════════════
print("\n── Model A: Sales Amount Predictor ──")
y_sales = df_model["Total_Sales_Amount"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y_sales, test_size=0.2, random_state=42)

rf_sales = RandomForestRegressor(n_estimators=200, max_depth=12,
                                  min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_sales.fit(X_tr, y_tr)
y_pred_sales = rf_sales.predict(X_te)

mae_s = mean_absolute_error(y_te, y_pred_sales)
r2_s  = r2_score(y_te, y_pred_sales)
cv_s  = cross_val_score(clone(rf_sales), X, y_sales, cv=5, scoring="r2").mean()
print(f"  MAE: {mae_s:,.0f}  |  R²: {r2_s:.4f}  |  CV R²: {cv_s:.4f}")
scores_log["Sales_Amount"] = {"MAE": round(mae_s, 2), "R2": round(r2_s, 4), "CV_R2": round(cv_s, 4)}

# ══════════════════════════════════════════════════════════
# 4. MODEL B – Net Profit Margin Predictor
# ══════════════════════════════════════════════════════════
print("\n── Model B: Net Profit Margin Predictor ──")
y_margin = df_model["Net_Profit_Margin"]
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_margin, test_size=0.2, random_state=42)

gb_margin = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, subsample=0.8,
                                       random_state=42)
gb_margin.fit(X_tr2, y_tr2)
y_pred_margin = gb_margin.predict(X_te2)

mae_m = mean_absolute_error(y_te2, y_pred_margin)
r2_m  = r2_score(y_te2, y_pred_margin)
cv_m  = cross_val_score(clone(gb_margin), X, y_margin, cv=5, scoring="r2").mean()
print(f"  MAE: {mae_m:.2f}%  |  R²: {r2_m:.4f}  |  CV R²: {cv_m:.4f}")
scores_log["Profit_Margin"] = {"MAE": round(mae_m, 4), "R2": round(r2_m, 4), "CV_R2": round(cv_m, 4)}

# ══════════════════════════════════════════════════════════
# 5. MODEL C – Sell Quantity Predictor
# ══════════════════════════════════════════════════════════
print("\n── Model C: Sell Quantity Predictor ──")
y_qty = df_model["Total_Qty_Sold"]
X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X, y_qty, test_size=0.2, random_state=42)

rf_qty = RandomForestRegressor(n_estimators=200, max_depth=10,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_qty.fit(X_tr3, y_tr3)
y_pred_qty = rf_qty.predict(X_te3)

mae_q = mean_absolute_error(y_te3, y_pred_qty)
r2_q  = r2_score(y_te3, y_pred_qty)
cv_q  = cross_val_score(clone(rf_qty), X, y_qty, cv=5, scoring="r2").mean()
print(f"  MAE: {mae_q:.1f} units  |  R²: {r2_q:.4f}  |  CV R²: {cv_q:.4f}")
scores_log["Sell_Quantity"] = {"MAE": round(mae_q, 2), "R2": round(r2_q, 4), "CV_R2": round(cv_q, 4)}

# ══════════════════════════════════════════════════════════
# 6. MODEL D – High / Low Profit Classifier
# ══════════════════════════════════════════════════════════
print("\n── Model D: Profit Class Classifier ──")

X_tr4, X_te4, y_margin_tr, y_margin_te = train_test_split(
    X, y_margin, test_size=0.2, random_state=42
)
threshold = y_margin_tr.median()
print(f"  Profit class threshold (train median): {threshold:.2f}%")

y_cls    = (y_margin >= threshold).astype(int)
y_cls_tr = (y_margin_tr >= threshold).astype(int)
y_cls_te = (y_margin_te >= threshold).astype(int)

rf_cls = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_cls.fit(X_tr4, y_cls_tr)
y_pred_cls = rf_cls.predict(X_te4)

acc = accuracy_score(y_cls_te, y_pred_cls)
print(f"  Accuracy: {acc:.4f}")
print(classification_report(y_cls_te, y_pred_cls,
                             target_names=["Low Profit", "High Profit"]))
scores_log["Profit_Classifier"] = {"Accuracy": round(acc, 4)}

df_model["Profit_Threshold"] = threshold

# ══════════════════════════════════════════════════════════
# 7. Predictions on full dataset
# ══════════════════════════════════════════════════════════
X_full = df_model[FEATURE_COLS].apply(lambda c: c.fillna(c.median()))

df_model["Predicted_Sales"]         = rf_sales.predict(X_full)
df_model["Predicted_Margin_Pct"]    = gb_margin.predict(X_full)
df_model["Predicted_Sell_Qty"]      = rf_qty.predict(X_full).round().astype(int)
df_model["Predicted_Profit_Class"]  = pd.Series(
    rf_cls.predict(X_full), index=df_model.index
).map({1: "High Profit", 0: "Low Profit"})

df_model["Predicted_Gross_Profit"]   = (
    df_model["Predicted_Sales"] * (df_model["Predicted_Margin_Pct"] / 100)
)
df_model["Predicted_Transport_Cost"] = (
    df_model["Avg_Unit_Cost"] * df_model["Predicted_Sell_Qty"] * 0.02
)
df_model["Predicted_Net_Profit"]     = (
    df_model["Predicted_Gross_Profit"] - df_model["Predicted_Transport_Cost"]
)

# ══════════════════════════════════════════════════════════
# 8. Monthly & Yearly summaries
# ══════════════════════════════════════════════════════════
monthly = (
    df_model.groupby(["Year", "MonthNum", "Region", "Category"])
    .agg(
        Monthly_Sales    = ("Total_Sales_Amount",   "sum"),
        Monthly_Profit   = ("Net_Profit",           "sum"),
        Monthly_Qty      = ("Total_Qty_Sold",       "sum"),
        Avg_Margin_Pct   = ("Net_Profit_Margin",    "mean"),
        Spot_Txn         = ("Retail_Txn",           "sum"),
        Online_Txn_Count = ("Online_Txn",           "sum"),
        Transport_Cost   = ("Transport_Cost_Est",   "sum"),
        Pred_Sales       = ("Predicted_Sales",      "sum"),
        Pred_Profit      = ("Predicted_Net_Profit", "sum"),
    )
    .reset_index()
)

yearly = (
    df_model.groupby(["Year", "Region", "Category"])
    .agg(
        Yearly_Sales     = ("Total_Sales_Amount",  "sum"),
        Yearly_Profit    = ("Net_Profit",          "sum"),
        Yearly_Qty       = ("Total_Qty_Sold",      "sum"),
        Avg_Margin_Pct   = ("Net_Profit_Margin",   "mean"),
        Total_Transport  = ("Transport_Cost_Est",  "sum"),
        Spot_Txn         = ("Retail_Txn",          "sum"),
        Online_Txn_Count = ("Online_Txn",          "sum"),
    )
    .reset_index()
)

# ══════════════════════════════════════════════════════════
# 9. Improvement Opportunities Report
# ══════════════════════════════════════════════════════════
top_quartile_margin = df_model["Net_Profit_Margin"].quantile(0.75)
insights = []

for (region, cat), grp in df_model.groupby(["Region", "Category"]):
    row = {}
    row["Region"]   = region
    row["Category"] = cat

    avg_margin = grp["Net_Profit_Margin"].mean()
    row["Avg_Margin_Pct"]                = round(avg_margin, 2)
    row["Margin_Gap_vs_Top_Quartile"]    = round(max(top_quartile_margin - avg_margin, 0), 2)
    row["Avg_Discount_Pct"]             = round(grp["Avg_Discount_Sales"].mean() * 100, 2)
    row["Avg_Price_vs_Competitor"]      = round(grp["Price_vs_Competitor"].mean(), 2)
    row["Avg_Inventory_Coverage_Days"]  = round(grp["Inventory_Coverage"].mean(), 1)
    row["Spot_Sales_Ratio_Avg"]         = round(grp["Spot_Sales_Ratio"].mean() * 100, 2)
    row["Online_Sales_Ratio_Avg"]       = round(grp["Online_Sales_Ratio"].mean() * 100, 2)
    row["Fulfillment_Rate_Avg"]         = round(grp["Fulfillment_Rate"].mean() * 100, 2)

    tips = []
    if row["Margin_Gap_vs_Top_Quartile"] > 5:
        tips.append(
            f"Margin is {row['Margin_Gap_vs_Top_Quartile']:.1f}pp below top performers – "
            "review pricing or reduce supplier costs."
        )
    if row["Avg_Discount_Pct"] > 20:
        tips.append(
            f"High avg discount ({row['Avg_Discount_Pct']:.1f}%) – "
            "consider loyalty rewards instead of blanket discounts."
        )
    if row["Avg_Price_vs_Competitor"] < -5:
        tips.append(
            "Priced below competitors – possible room to raise prices without volume loss."
        )
    if row["Avg_Inventory_Coverage_Days"] > 60:
        tips.append(
            "Over-stocked (>60 days coverage) – reduce order quantities to free up working capital."
        )
    if row["Avg_Inventory_Coverage_Days"] < 10:
        tips.append(
            "Under-stocked (<10 days coverage) – risk of stockouts; increase re-order frequency."
        )
    if row["Fulfillment_Rate_Avg"] < 70:
        tips.append(
            "Low demand fulfilment – consider stocking more to capture latent demand."
        )
    if row["Online_Sales_Ratio_Avg"] < 20:
        tips.append(
            "Low online sales share – invest in e-commerce / marketplace presence."
        )
    row["Recommendations"] = " | ".join(tips) if tips else "Performance is on-track."
    insights.append(row)

improvement_df = pd.DataFrame(insights)

# ══════════════════════════════════════════════════════════
# 10. Feature Importance
# ══════════════════════════════════════════════════════════
fi_df = pd.DataFrame({
    "Feature"          : FEATURE_COLS,
    "Importance_Sales" : rf_sales.feature_importances_,
    "Importance_Margin": gb_margin.feature_importances_,
    "Importance_Qty"   : rf_qty.feature_importances_,
    "Importance_Cls"   : rf_cls.feature_importances_,
}).sort_values("Importance_Sales", ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════
# 11. FUTURE PREDICTIONS
# ══════════════════════════════════════════════════════════
print("\n── Generating 12-month Future Predictions ──")

def generate_future_predictions(df_hist, label_encoders, feature_cols,
                                 rf_sales, gb_margin, rf_qty, rf_cls,
                                 threshold, n_months=12):
    last_month   = df_hist["Month"].max()
    regions      = df_hist["Region"].unique()
    categories   = df_hist["Category"].unique()

    future_rows = []

    for region in regions:
        for cat in categories:
            seg = df_hist[
                (df_hist["Region"] == region) &
                (df_hist["Category"] == cat)
            ].sort_values("Month").copy()

            if len(seg) < 3:
                continue

            recent = seg.tail(12)

            # Monthly averages keyed by month number
            monthly_avg = {}
            for mn, grp in recent.groupby("MonthNum"):
                monthly_avg[mn] = {
                    "Total_Sales_Amount": grp["Total_Sales_Amount"].mean(),
                    "Net_Profit_Margin":  grp["Net_Profit_Margin"].mean(),
                    "Total_Qty_Sold":     grp["Total_Qty_Sold"].mean(),
                }

            x_idx = np.arange(len(seg))
            def _slope(series):
                if series.std() == 0 or len(series) < 2:
                    return 0.0
                return np.polyfit(x_idx, series.values, 1)[0]

            slope_sales  = _slope(seg["Total_Sales_Amount"])
            slope_margin = _slope(seg["Net_Profit_Margin"])
            slope_qty    = _slope(seg["Total_Qty_Sold"])

            template = recent.iloc[-1].copy()

            for i in range(1, n_months + 1):
                future_month = last_month + pd.DateOffset(months=i)
                month_num    = future_month.month
                year         = future_month.year
                quarter      = (month_num - 1) // 3 + 1

                # Seasonal base — fall back to recent mean if month not in history
                base = monthly_avg.get(month_num, {
                    "Total_Sales_Amount": recent["Total_Sales_Amount"].mean(),
                    "Net_Profit_Margin":  recent["Net_Profit_Margin"].mean(),
                    "Total_Qty_Sold":     recent["Total_Qty_Sold"].mean(),
                })
                base_sales  = base["Total_Sales_Amount"]
                base_margin = base["Net_Profit_Margin"]
                base_qty    = base["Total_Qty_Sold"]

                offset     = len(seg) + i
                exp_sales  = max(base_sales  + slope_sales  * (offset - x_idx.mean()), 0)
                exp_margin = base_margin + slope_margin * (offset - x_idx.mean())
                exp_qty    = max(base_qty + slope_qty * (offset - x_idx.mean()), 0)

                feat_row = {
                    "Region_enc"           : label_encoders["Region"].transform([region])[0],
                    "Category_enc"         : label_encoders["Category"].transform([cat])[0],
                    "Year"                 : year,
                    "MonthNum"             : month_num,
                    "Quarter"              : quarter,
                    "Avg_Unit_Cost"        : template.get("Avg_Unit_Cost",       0),
                    "Avg_Discount_Sales"   : template.get("Avg_Discount_Sales",  0),
                    "Num_Transactions"     : template.get("Num_Transactions",    0),
                    "New_Customers"        : template.get("New_Customers",       0),
                    "Returning_Customers"  : template.get("Returning_Customers", 0),
                    "Online_Txn"           : template.get("Online_Txn",          0),
                    "Retail_Txn"           : template.get("Retail_Txn",          0),
                    "Inventory_Level"      : template.get("Inventory_Level",     0),
                    "Units_Ordered"        : template.get("Units_Ordered",       0),
                    "Demand_Forecast"      : exp_qty * 1.10,
                    "Avg_Discount_Inv"     : template.get("Avg_Discount_Inv",    0),
                    "Holiday_Days"         : template.get("Holiday_Days",        0),
                    "Competitor_Pricing"   : template.get("Competitor_Pricing",  0),
                    "Seasonality_enc"      : template.get("Seasonality_enc",     0),
                    "Weather_Mode_enc"     : template.get("Weather_Mode_enc",    0),
                    "Price_vs_Competitor"  : template.get("Price_vs_Competitor", 0),
                    "Fulfillment_Rate"     : template.get("Fulfillment_Rate",    1),
                    "Recommended_Order_Qty": exp_qty * 1.15,
                }

                feat_vals = pd.DataFrame([feat_row])[feature_cols]

                ml_sales  = float(rf_sales.predict(feat_vals)[0])
                ml_margin = float(gb_margin.predict(feat_vals)[0])
                ml_qty    = float(rf_qty.predict(feat_vals)[0])
                ml_cls    = int(rf_cls.predict(feat_vals)[0])

                f_sales  = 0.6 * ml_sales  + 0.4 * exp_sales
                f_margin = 0.6 * ml_margin + 0.4 * exp_margin
                f_qty    = max(0, 0.6 * ml_qty + 0.4 * exp_qty)

                sales_std = recent["Total_Sales_Amount"].std()
                conf_band = sales_std * np.sqrt(i)

                avg_unit_cost = template.get("Avg_Unit_Cost", 0)
                f_net_profit  = f_sales * (f_margin / 100) - avg_unit_cost * f_qty * 0.02

                future_rows.append({
                    "Month"                : future_month,
                    "Region"               : region,
                    "Category"             : cat,
                    "Forecast_Sales"       : round(f_sales,  2),
                    "Forecast_Margin_Pct"  : round(f_margin, 2),
                    "Forecast_Qty"         : int(round(f_qty)),
                    "Forecast_Net_Profit"  : round(f_net_profit, 2),
                    "Forecast_Profit_Class": "High Profit" if ml_cls == 1 else "Low Profit",
                    "Forecast_Lower_Sales" : round(max(0, f_sales - conf_band), 2),
                    "Forecast_Upper_Sales" : round(f_sales + conf_band, 2),
                    "Months_Ahead"         : i,
                })

    return pd.DataFrame(future_rows)


future_df = generate_future_predictions(
    df_hist        = df_model,
    label_encoders = label_encoders,
    feature_cols   = FEATURE_COLS,
    rf_sales       = rf_sales,
    gb_margin      = gb_margin,
    rf_qty         = rf_qty,
    rf_cls         = rf_cls,
    threshold      = threshold,
    n_months       = 12,
)

print(f"  Future prediction rows: {len(future_df)}")
print(f"  Forecast period: {future_df['Month'].min().strftime('%b %Y')} → {future_df['Month'].max().strftime('%b %Y')}")

# ══════════════════════════════════════════════════════════
# 12. Save all outputs
# ══════════════════════════════════════════════════════════
PRED_COLS = [
    "Month", "Region", "Category",
    "Total_Sales_Amount", "Total_Qty_Sold",
    "Avg_Unit_Cost", "Avg_Unit_Price",
    "Gross_Profit", "Transport_Cost_Est", "Net_Profit",
    "Net_Profit_Margin", "Profit_Margin_Pct",
    "Spot_Sales_Ratio", "Online_Sales_Ratio",
    "Inventory_Level", "Inventory_Coverage",
    "Demand_Forecast", "Recommended_Price", "Recommended_Order_Qty",
    "Predicted_Sales", "Predicted_Margin_Pct",
    "Predicted_Sell_Qty", "Predicted_Net_Profit",
    "Predicted_Profit_Class",
    "Sales_Growth_MoM", "Profit_Growth_MoM",
    "Fulfillment_Rate",
]
PRED_COLS = [c for c in PRED_COLS if c in df_model.columns]

df_model[PRED_COLS].to_csv(f"{PROJECT_DIR}/predictions.csv",         index=False)
monthly.to_csv(            f"{PROJECT_DIR}/monthly_summary.csv",      index=False)
yearly.to_csv(             f"{PROJECT_DIR}/yearly_summary.csv",       index=False)
improvement_df.to_csv(     f"{PROJECT_DIR}/improvement_report.csv",   index=False)
fi_df.to_csv(              f"{PROJECT_DIR}/feature_importance.csv",   index=False)
future_df.to_csv(          f"{PROJECT_DIR}/future_predictions.csv",   index=False)
pd.DataFrame(scores_log).T.reset_index().rename(
    columns={"index": "Model"}
).to_csv(f"{PROJECT_DIR}/model_scores.csv", index=False)

print("\n✅  All outputs saved:")
for f in ["predictions.csv", "monthly_summary.csv", "yearly_summary.csv",
          "improvement_report.csv", "feature_importance.csv",
          "model_scores.csv", "future_predictions.csv"]:
    path = f"{PROJECT_DIR}/{f}"
    if os.path.exists(path):
        rows = len(pd.read_csv(path))
        print(f"   {path}  ({rows} rows)")