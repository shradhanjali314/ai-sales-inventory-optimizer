"""
============================================================
  APP.PY  –  Smart Retail Intelligence  |  Streamlit UI
  Run:  streamlit run app.py
  Requires: predictions.csv, monthly_summary.csv,
            yearly_summary.csv, improvement_report.csv,
            feature_importance.csv, model_scores.csv,
            future_predictions.csv
            (all inside ./smart_retail_ml/)
============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="BizzInsight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    section[data-testid="stSidebar"] { background-color: #181b24; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    [data-testid="stMetric"] {
        background: #181b24;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"]  { color: #7c8099 !important; font-size: 12px !important; }
    [data-testid="stMetricValue"]  { color: #e8eaf0 !important; }
    [data-testid="stMetricDelta"]  { font-size: 12px !important; }

    .stTabs [data-baseweb="tab-list"]  { gap: 6px; background: #181b24; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"]       { background: transparent; color: #7c8099; border-radius: 6px; padding: 6px 16px; }
    .stTabs [aria-selected="true"]     { background: #1e2130 !important; color: #6c8ef7 !important; }

    .stDataFrame { border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; }
    h1, h2, h3 { color: #e8eaf0 !important; }
    .js-plotly-plot { border-radius: 10px; }

    .forecast-badge {
        display: inline-block;
        background: rgba(108,142,247,0.15);
        border: 1px solid rgba(108,142,247,0.4);
        color: #6c8ef7;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: .5px;
        margin-left: 8px;
        vertical-align: middle;
    }
    .custom-year-badge {
        display: inline-block;
        background: rgba(176,106,247,0.15);
        border: 1px solid rgba(176,106,247,0.4);
        color: #b06af7;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: .5px;
        margin-left: 8px;
        vertical-align: middle;
    }
    .info-box {
        background: #1e2130;
        border: 1px solid rgba(108,142,247,0.2);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 16px;
        font-size: 13px;
        color: #9ca0b0;
    }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────
COLORS = {
    "accent": "#6c8ef7",
    "green":  "#3dd68c",
    "amber":  "#f0a842",
    "red":    "#f06b6b",
    "teal":   "#38c9b8",
    "muted":  "#7c8099",
    "purple": "#b06af7",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#7c8099", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7c8099")),
)

CATEGORY_COLORS = {
    "Clothing":    COLORS["accent"],
    "Electronics": COLORS["teal"],
    "Food":        COLORS["green"],
    "Furniture":   COLORS["amber"],
}

REGION_COLORS = {
    "East":  COLORS["accent"],
    "North": COLORS["green"],
    "South": COLORS["amber"],
    "West":  COLORS["teal"],
}

# ── Data loading ──────────────────────────────────────────
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smart_retail_ml")

@st.cache_data(show_spinner="Loading data…")
def load_data():
    def read(name):
        path = os.path.join(PROJECT_DIR, name)
        if not os.path.exists(path):
            st.error(f"❌ File not found: {path}\n\nPlease run build_dataset.py then ml_pipeline.py first.")
            st.stop()
        return pd.read_csv(path)

    preds   = read("predictions.csv")
    monthly = read("monthly_summary.csv")
    yearly  = read("yearly_summary.csv")
    improve = read("improvement_report.csv")
    fi      = read("feature_importance.csv")
    scores  = read("model_scores.csv")
    future  = read("future_predictions.csv")

    preds["Month"]  = pd.to_datetime(preds["Month"])
    future["Month"] = pd.to_datetime(future["Month"])

    preds["Year"]       = preds["Month"].dt.year
    preds["MonthNum"]   = preds["Month"].dt.month
    preds["MonthLabel"] = preds["Month"].dt.strftime("%b %Y")

    future["Year"]       = future["Month"].dt.year
    future["MonthNum"]   = future["Month"].dt.month
    future["MonthLabel"] = future["Month"].dt.strftime("%b %Y")

    return preds, monthly, yearly, improve, fi, scores, future

preds, monthly, yearly, improve, fi_df, scores_df, future = load_data()

# ── Custom year prediction helper ─────────────────────────
def predict_for_year(preds_df, target_year, sel_regions, sel_cats):
    """
    Generate full-year (Jan–Dec) predictions for any target_year:
    - Uses actual recorded data where it exists for that year.
    - Extrapolates from seasonal averages + linear trend for missing months
      or future years entirely.
    Returns a DataFrame compatible with the future_predictions schema.
    """
    month_name = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    rows = []

    for region in sel_regions:
        for cat in sel_cats:
            seg = preds_df[
                (preds_df["Region"] == region) &
                (preds_df["Category"] == cat)
            ].sort_values("Month").copy()

            if seg.empty:
                continue

            actual_yr = seg[seg["Year"] == target_year]

            # Historical data to base the trend/seasonality on
            hist = seg[seg["Year"] < target_year]
            if hist.empty:
                hist = seg  # if target_year <= min year, use all data

            x_idx = np.arange(len(hist))

            def _slope(col):
                vals = hist[col].values
                if len(vals) < 2 or np.std(vals) == 0:
                    return 0.0
                return float(np.polyfit(x_idx, vals, 1)[0])

            slope_sales  = _slope("Total_Sales_Amount")
            slope_margin = _slope("Net_Profit_Margin")
            slope_qty    = _slope("Total_Qty_Sold")

            # Monthly seasonal averages from history
            monthly_avg = {}
            for mn, grp in hist.groupby("MonthNum"):
                monthly_avg[mn] = {
                    "sales":  grp["Total_Sales_Amount"].mean(),
                    "margin": grp["Net_Profit_Margin"].mean(),
                    "qty":    grp["Total_Qty_Sold"].mean(),
                }

            sales_std    = seg["Total_Sales_Amount"].std()
            margin_med   = seg["Net_Profit_Margin"].median()
            avg_cost     = seg["Avg_Unit_Cost"].mean() if "Avg_Unit_Cost" in seg.columns else 0

            for month_num in range(1, 13):
                target_date = pd.Timestamp(year=target_year, month=month_num, day=1)
                label       = f"{month_name[month_num - 1]} {target_year}"

                # Check for actual recorded data first
                actual_row = actual_yr[actual_yr["MonthNum"] == month_num]
                if not actual_row.empty:
                    r        = actual_row.iloc[0]
                    f_sales  = float(r["Total_Sales_Amount"])
                    f_margin = float(r["Net_Profit_Margin"])
                    f_qty    = float(r["Total_Qty_Sold"])
                    f_profit = float(r["Net_Profit"])
                    is_actual = True
                    conf_band = 0.0
                else:
                    # Seasonal base
                    base = monthly_avg.get(month_num, {
                        "sales":  hist["Total_Sales_Amount"].mean(),
                        "margin": hist["Net_Profit_Margin"].mean(),
                        "qty":    hist["Total_Qty_Sold"].mean(),
                    })

                    # How many steps ahead from end of history?
                    years_ahead  = target_year - int(hist["Year"].max())
                    month_offset = (years_ahead - 1) * 12 + month_num
                    steps        = len(hist) + max(month_offset, 1)
                    delta        = steps - x_idx.mean()

                    f_sales  = max(base["sales"]  + slope_sales  * delta, 0)
                    f_margin = base["margin"] + slope_margin * delta
                    f_qty    = max(base["qty"]    + slope_qty    * delta, 0)
                    f_profit = f_sales * (f_margin / 100) - avg_cost * f_qty * 0.02
                    is_actual = False
                    conf_band = sales_std * np.sqrt(max(month_offset, 1))

                rows.append({
                    "Month"                : target_date,
                    "MonthLabel"           : label,
                    "MonthNum"             : month_num,
                    "Region"               : region,
                    "Category"             : cat,
                    "Forecast_Sales"       : round(f_sales,  2),
                    "Forecast_Margin_Pct"  : round(f_margin, 2),
                    "Forecast_Qty"         : int(round(f_qty)),
                    "Forecast_Net_Profit"  : round(f_profit, 2),
                    "Forecast_Profit_Class": "High Profit" if f_margin >= margin_med else "Low Profit",
                    "Forecast_Lower_Sales" : round(max(0, f_sales - conf_band), 2),
                    "Forecast_Upper_Sales" : round(f_sales + conf_band, 2),
                    "Is_Actual"            : is_actual,
                })

    return pd.DataFrame(rows)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")

    years = sorted(preds["Year"].unique())
    sel_year = st.selectbox("Year", options=["All"] + [str(y) for y in years], index=0)

    regions = sorted(preds["Region"].unique())
    sel_regions = st.multiselect("Region", options=regions, default=regions)

    categories = sorted(preds["Category"].unique())
    sel_cats = st.multiselect("Category", options=categories, default=categories)

    st.markdown("---")
    st.markdown("### 🔭 Forecast Settings")
    forecast_months = st.slider("Next-N Horizon (months)", min_value=1, max_value=12, value=12)

    st.markdown("---")
    st.markdown("### 📅 Custom Year Prediction")

    min_yr = int(preds["Year"].min())
    max_yr = int(preds["Year"].max())

    custom_year = st.number_input(
        "Enter any year to predict",
        min_value=min_yr,
        max_value=max_yr + 10,
        value=max_yr + 1,
        step=1,
        help=f"Historical data covers {min_yr}–{max_yr}. Years beyond {max_yr} will be extrapolated.",
    )
    run_custom = st.button("🚀 Generate Year Prediction", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#7c8099;'>"
        "BizzInsight AI · Smart Retail<br>"
        "Powered by RandomForest & GradientBoosting"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Apply filters ─────────────────────────────────────────
mask = preds["Region"].isin(sel_regions) & preds["Category"].isin(sel_cats)
if sel_year != "All":
    mask &= preds["Year"] == int(sel_year)
df_f = preds[mask].copy()

m_mask = monthly["Region"].isin(sel_regions) & monthly["Category"].isin(sel_cats)
if sel_year != "All":
    m_mask &= monthly["Year"] == int(sel_year)
monthly_f = monthly[m_mask].copy()

future_f = future[
    future["Region"].isin(sel_regions) &
    future["Category"].isin(sel_cats) &
    (future["Months_Ahead"] <= forecast_months)
].copy()

if df_f.empty:
    st.warning("No data matches the selected filters. Please adjust the sidebar.")
    st.stop()

# ── Session state for custom year ─────────────────────────
if "custom_year_df"  not in st.session_state:
    st.session_state.custom_year_df  = None
if "custom_year_val" not in st.session_state:
    st.session_state.custom_year_val = None

if run_custom:
    with st.spinner(f"Generating predictions for {custom_year}…"):
        st.session_state.custom_year_df  = predict_for_year(preds, custom_year, sel_regions, sel_cats)
        st.session_state.custom_year_val = custom_year

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab_overview, tab_pred, tab_forecast, tab_custom, tab_improve, tab_models = st.tabs([
    "📈 Overview", "🔮 Predictions", "🔭 Future Forecast",
    "📅 Custom Year", "💡 Improvements", "🤖 ML Models",
])

# ─────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────────────
with tab_overview:
    total_rev    = df_f["Total_Sales_Amount"].sum()
    total_profit = df_f["Net_Profit"].sum()
    avg_margin   = df_f["Net_Profit_Margin"].mean()
    total_qty    = df_f["Total_Qty_Sold"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Total Revenue",    f"₹{total_rev/1e5:,.1f}L")
    k2.metric("📦 Net Profit",        f"₹{total_profit/1e5:,.1f}L")
    k3.metric("📊 Avg Profit Margin", f"{avg_margin:.1f}%")
    k4.metric("🛒 Units Sold",        f"{int(total_qty):,}")

    st.markdown("---")

    monthly_trend = (
        monthly_f.groupby("MonthNum", as_index=False)
        .agg(Revenue=("Monthly_Sales", "sum"), Profit=("Monthly_Profit", "sum"))
        .sort_values("MonthNum")
    )
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_trend["Month"] = monthly_trend["MonthNum"].apply(
        lambda x: month_labels[x - 1] if 1 <= x <= 12 else str(x)
    )

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(x=monthly_trend["Month"], y=monthly_trend["Revenue"],
               name="Revenue", marker_color=COLORS["accent"], marker_opacity=0.7),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Scatter(x=monthly_trend["Month"], y=monthly_trend["Profit"],
                   name="Profit", line=dict(color=COLORS["green"], width=2.5),
                   fill="tozeroy", fillcolor="rgba(61,214,140,0.08)",
                   mode="lines+markers", marker=dict(size=5)),
        secondary_y=False,
    )
    fig_trend.update_layout(title="Monthly Revenue vs Profit", **PLOTLY_LAYOUT,
                             yaxis=dict(tickprefix="₹", tickformat=".2s"), barmode="overlay")
    st.plotly_chart(fig_trend, use_container_width=True)

    col_l, col_r = st.columns(2)

    cat_sales = df_f.groupby("Category", as_index=False)["Total_Sales_Amount"].sum()
    fig_cat = px.pie(cat_sales, names="Category", values="Total_Sales_Amount",
                     hole=0.6, color="Category", color_discrete_map=CATEGORY_COLORS,
                     title="Sales by Category")
    fig_cat.update_traces(textinfo="label+percent")
    fig_cat.update_layout(**PLOTLY_LAYOUT)
    col_l.plotly_chart(fig_cat, use_container_width=True)

    reg_sales = df_f.groupby("Region", as_index=False)["Total_Sales_Amount"].sum().sort_values("Total_Sales_Amount")
    fig_reg = px.bar(reg_sales, x="Total_Sales_Amount", y="Region", orientation="h",
                     color="Region", color_discrete_map=REGION_COLORS, title="Revenue by Region")
    fig_reg.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                           xaxis=dict(tickprefix="₹", tickformat=".2s"))
    col_r.plotly_chart(fig_reg, use_container_width=True)

    ch = df_f.groupby("Category", as_index=False).agg(
        Spot=("Spot_Sales_Ratio", "mean"), Online=("Online_Sales_Ratio", "mean"))
    ch["Spot"]   *= 100
    ch["Online"] *= 100
    fig_ch = go.Figure()
    fig_ch.add_bar(x=ch["Category"], y=ch["Spot"],   name="Retail/Spot", marker_color=COLORS["amber"])
    fig_ch.add_bar(x=ch["Category"], y=ch["Online"], name="Online",      marker_color=COLORS["accent"])
    fig_ch.update_layout(barmode="stack", title="Spot vs Online Channel Split by Category (%)",
                          **PLOTLY_LAYOUT, yaxis=dict(title="% of Transactions", range=[0, 100]))
    st.plotly_chart(fig_ch, use_container_width=True)

# ─────────────────────────────────────────────────────────
# TAB 2 · PREDICTIONS
# ─────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("### Model Predictions by Segment")

    pred_agg = (
        df_f.groupby(["Region", "Category"], as_index=False)
        .agg(Actual_Sales=("Total_Sales_Amount","sum"), Predicted_Sales=("Predicted_Sales","sum"),
             Net_Profit=("Net_Profit","sum"), Avg_Margin=("Net_Profit_Margin","mean"),
             Pred_Sell_Qty=("Predicted_Sell_Qty","sum"), Transport_Cost=("Transport_Cost_Est","sum"))
        .round(2)
    )
    cls_mode = (df_f.groupby(["Region","Category"])["Predicted_Profit_Class"]
                .agg(lambda x: x.mode().iloc[0]).reset_index())
    pred_agg = pred_agg.merge(cls_mode, on=["Region","Category"], how="left")
    pred_agg.rename(columns={"Predicted_Profit_Class": "Profit_Class"}, inplace=True)

    fig_pa = px.scatter(pred_agg, x="Actual_Sales", y="Predicted_Sales",
                        color="Category", symbol="Region", color_discrete_map=CATEGORY_COLORS,
                        size=[20]*len(pred_agg), title="Predicted vs Actual Sales by Segment",
                        hover_data=["Region","Category","Avg_Margin"])
    max_val = max(pred_agg["Actual_Sales"].max(), pred_agg["Predicted_Sales"].max())
    fig_pa.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                     line=dict(color=COLORS["muted"], dash="dot", width=1))
    fig_pa.update_layout(**PLOTLY_LAYOUT, xaxis=dict(tickprefix="₹", tickformat=".2s"),
                          yaxis=dict(tickprefix="₹", tickformat=".2s"))
    st.plotly_chart(fig_pa, use_container_width=True)

    month_order = df_f[["Month","MonthLabel"]].drop_duplicates().sort_values("Month")["MonthLabel"].tolist()
    monthly_pred = (df_f.groupby("MonthLabel", as_index=False)
                    .agg(Actual=("Total_Sales_Amount","sum"), Predicted=("Predicted_Sales","sum")))
    monthly_pred = monthly_pred.set_index("MonthLabel").reindex(month_order).reset_index()

    fig_mp = go.Figure()
    fig_mp.add_trace(go.Scatter(x=monthly_pred["MonthLabel"], y=monthly_pred["Actual"],
                                name="Actual", line=dict(color=COLORS["accent"], width=2), mode="lines+markers"))
    fig_mp.add_trace(go.Scatter(x=monthly_pred["MonthLabel"], y=monthly_pred["Predicted"],
                                name="Predicted", line=dict(color=COLORS["amber"], width=2, dash="dot"),
                                mode="lines+markers"))
    fig_mp.update_layout(title="Actual vs Predicted Monthly Sales", **PLOTLY_LAYOUT,
                          yaxis=dict(tickprefix="₹", tickformat=".2s"))
    st.plotly_chart(fig_mp, use_container_width=True)

    st.markdown("#### Segment Prediction Table")
    display_df = pred_agg.copy()
    for col in ["Actual_Sales","Predicted_Sales","Net_Profit","Transport_Cost"]:
        display_df[col] = display_df[col].apply(lambda v: f"₹{v:,.0f}")
    display_df["Avg_Margin"]    = display_df["Avg_Margin"].apply(lambda v: f"{v:.1f}%")
    display_df["Pred_Sell_Qty"] = display_df["Pred_Sell_Qty"].apply(lambda v: f"{int(v):,} units")

    def highlight_class(val):
        if val == "High Profit":
            return "background-color: rgba(61,214,140,0.15); color: #3dd68c; font-weight:600"
        return "background-color: rgba(240,168,66,0.15); color: #f0a842; font-weight:600"

    st.dataframe(display_df.style.map(highlight_class, subset=["Profit_Class"]),
                 use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────
# TAB 3 · FUTURE FORECAST
# ─────────────────────────────────────────────────────────
with tab_forecast:
    st.markdown(
        "### 🔭 Future Sales Forecast"
        "<span class='forecast-badge'>12-Month Ahead</span>",
        unsafe_allow_html=True,
    )
    st.caption(f"Showing **{forecast_months}-month** forecast · Shaded band = ±1 std confidence interval")

    if future_f.empty:
        st.warning("No future predictions available for the selected filters.")
        st.stop()

    fk1, fk2, fk3, fk4 = st.columns(4)
    fk1.metric("📅 Forecast Period",
               f"{future_f['Month'].min().strftime('%b %Y')} → {future_f['Month'].max().strftime('%b %Y')}")
    fk2.metric("🔮 Forecast Revenue",    f"₹{future_f['Forecast_Sales'].sum()/1e5:,.1f}L")
    fk3.metric("💹 Avg Forecast Margin", f"{future_f['Forecast_Margin_Pct'].mean():.1f}%")
    fk4.metric("📦 Forecast Units",      f"{int(future_f['Forecast_Qty'].sum()):,}")
    st.markdown("---")

    hist_line = df_f.groupby("Month", as_index=False)["Total_Sales_Amount"].sum().sort_values("Month")
    hist_line["MonthLabel"] = hist_line["Month"].dt.strftime("%b %Y")

    fut_line = (future_f.groupby("Month", as_index=False)
                .agg(Forecast_Sales=("Forecast_Sales","sum"),
                     Forecast_Lower_Sales=("Forecast_Lower_Sales","sum"),
                     Forecast_Upper_Sales=("Forecast_Upper_Sales","sum"))
                .sort_values("Month"))
    fut_line["MonthLabel"] = fut_line["Month"].dt.strftime("%b %Y")

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([fut_line["MonthLabel"], fut_line["MonthLabel"].iloc[::-1]]),
        y=pd.concat([fut_line["Forecast_Upper_Sales"], fut_line["Forecast_Lower_Sales"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(108,142,247,0.12)",
        line=dict(color="rgba(255,255,255,0)"), showlegend=True, name="Confidence Band",
    ))
    fig_fc.add_trace(go.Scatter(x=hist_line["MonthLabel"], y=hist_line["Total_Sales_Amount"],
                                name="Historical Sales", mode="lines+markers",
                                line=dict(color=COLORS["accent"], width=2.5), marker=dict(size=5)))
    fig_fc.add_trace(go.Scatter(x=fut_line["MonthLabel"], y=fut_line["Forecast_Sales"],
                                name="Forecast Sales", mode="lines+markers",
                                line=dict(color=COLORS["purple"], width=2.5, dash="dot"),
                                marker=dict(size=6, symbol="diamond")))
    last_hist = hist_line["MonthLabel"].iloc[-1]
    upper_y   = fut_line["Forecast_Upper_Sales"].max()
    fig_fc.add_trace(go.Scatter(x=[last_hist, last_hist], y=[0, upper_y], mode="lines",
                                line=dict(color=COLORS["muted"], dash="dash", width=1),
                                name="Forecast Start", showlegend=True))
    fig_fc.update_layout(title="Historical Sales + 12-Month Forecast", **PLOTLY_LAYOUT,
                          yaxis=dict(tickprefix="₹", tickformat=".2s"), xaxis=dict(tickangle=-35))
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("#### Forecast Revenue by Category")
    fut_cat = (future_f.groupby(["MonthLabel","Month","Category"], as_index=False)["Forecast_Sales"].sum()
               .sort_values("Month"))
    fig_fcat = px.line(fut_cat, x="MonthLabel", y="Forecast_Sales", color="Category",
                       color_discrete_map=CATEGORY_COLORS, markers=True,
                       title="Forecast Sales by Category")
    fig_fcat.update_layout(**PLOTLY_LAYOUT, yaxis=dict(tickprefix="₹", tickformat=".2s"),
                            xaxis=dict(tickangle=-35))
    st.plotly_chart(fig_fcat, use_container_width=True)

    col_reg, col_margin = st.columns(2)
    fut_reg = (future_f.groupby(["MonthLabel","Month","Region"], as_index=False)["Forecast_Sales"].sum()
               .sort_values("Month"))
    fig_freg = px.area(fut_reg, x="MonthLabel", y="Forecast_Sales", color="Region",
                       color_discrete_map=REGION_COLORS, title="Forecast Sales by Region (stacked area)")
    fig_freg.update_layout(**PLOTLY_LAYOUT, yaxis=dict(tickprefix="₹", tickformat=".2s"),
                            xaxis=dict(tickangle=-35))
    col_reg.plotly_chart(fig_freg, use_container_width=True)

    fut_margin = (future_f.groupby(["MonthLabel","Month"], as_index=False)["Forecast_Margin_Pct"].mean()
                  .sort_values("Month"))
    fig_fmarg = go.Figure()
    fig_fmarg.add_trace(go.Scatter(x=fut_margin["MonthLabel"], y=fut_margin["Forecast_Margin_Pct"],
                                   name="Avg Forecast Margin", mode="lines+markers",
                                   line=dict(color=COLORS["green"], width=2.5),
                                   fill="tozeroy", fillcolor="rgba(61,214,140,0.08)", marker=dict(size=6)))
    fig_fmarg.update_layout(title="Forecast Avg Net Profit Margin %", **PLOTLY_LAYOUT,
                             yaxis=dict(title="Margin %"), xaxis=dict(tickangle=-35))
    col_margin.plotly_chart(fig_fmarg, use_container_width=True)

    st.markdown("#### Segment Profit Classification — Forecast Period")
    col_d1, col_d2 = st.columns(2)
    cls_count = future_f.groupby("Forecast_Profit_Class", as_index=False)["Forecast_Sales"].sum()
    fig_cls = px.pie(cls_count, names="Forecast_Profit_Class", values="Forecast_Sales", hole=0.55,
                     color="Forecast_Profit_Class",
                     color_discrete_map={"High Profit": COLORS["green"], "Low Profit": COLORS["red"]},
                     title="Forecast Sales by Profit Class")
    fig_cls.update_traces(textinfo="label+percent")
    fig_cls.update_layout(**PLOTLY_LAYOUT)
    col_d1.plotly_chart(fig_cls, use_container_width=True)

    heat_data = (future_f.groupby(["Region","Category"])["Forecast_Sales"].sum().reset_index()
                 .pivot(index="Region", columns="Category", values="Forecast_Sales").fillna(0))
    fig_heat = px.imshow(heat_data, color_continuous_scale=["#181b24","#6c8ef7"], text_auto=".2s",
                         title="Forecast Revenue Heatmap (Region × Category)", aspect="auto")
    fig_heat.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, xaxis_title="", yaxis_title="")
    col_d2.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### Full Forecast Table")
    disp_fut = future_f[["Month","Region","Category","Forecast_Sales","Forecast_Margin_Pct",
                          "Forecast_Qty","Forecast_Net_Profit","Forecast_Profit_Class",
                          "Forecast_Lower_Sales","Forecast_Upper_Sales","Months_Ahead"]].copy()
    disp_fut["Month"]                = disp_fut["Month"].dt.strftime("%b %Y")
    disp_fut["Forecast_Sales"]       = disp_fut["Forecast_Sales"].apply(lambda v: f"₹{v:,.0f}")
    disp_fut["Forecast_Net_Profit"]  = disp_fut["Forecast_Net_Profit"].apply(lambda v: f"₹{v:,.0f}")
    disp_fut["Forecast_Lower_Sales"] = disp_fut["Forecast_Lower_Sales"].apply(lambda v: f"₹{v:,.0f}")
    disp_fut["Forecast_Upper_Sales"] = disp_fut["Forecast_Upper_Sales"].apply(lambda v: f"₹{v:,.0f}")
    disp_fut["Forecast_Margin_Pct"]  = disp_fut["Forecast_Margin_Pct"].apply(lambda v: f"{v:.1f}%")
    disp_fut["Forecast_Qty"]         = disp_fut["Forecast_Qty"].apply(lambda v: f"{v:,}")
    disp_fut.rename(columns={"Forecast_Sales":"Fcst Revenue","Forecast_Margin_Pct":"Fcst Margin",
                              "Forecast_Qty":"Fcst Units","Forecast_Net_Profit":"Fcst Net Profit",
                              "Forecast_Profit_Class":"Profit Class","Forecast_Lower_Sales":"Lower Bound",
                              "Forecast_Upper_Sales":"Upper Bound","Months_Ahead":"Months Ahead"}, inplace=True)

    def highlight_class_fut(val):
        if val == "High Profit":
            return "background-color: rgba(61,214,140,0.15); color: #3dd68c; font-weight:600"
        return "background-color: rgba(240,107,107,0.15); color: #f06b6b; font-weight:600"

    st.dataframe(disp_fut.style.map(highlight_class_fut, subset=["Profit Class"]),
                 use_container_width=True, hide_index=True)

    csv_bytes = future_f.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full Forecast CSV", data=csv_bytes,
                       file_name="future_predictions.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────
# TAB 4 · CUSTOM YEAR PREDICTION  ← NEW
# ─────────────────────────────────────────────────────────
with tab_custom:
    cy_df  = st.session_state.custom_year_df
    cy_val = st.session_state.custom_year_val

    if cy_df is None or cy_df.empty:
        st.markdown(
            "### 📅 Custom Year Prediction"
            "<span class='custom-year-badge'>Enter any year</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='info-box'>"
            "👈 Use the <b>Custom Year Prediction</b> panel in the sidebar:<br><br>"
            "1. Enter any year in the number box (historical or future).<br>"
            "2. Click <b>🚀 Generate Year Prediction</b>.<br>"
            "3. Results will appear here instantly.<br><br>"
            "• Years <b>within your dataset</b> show actual recorded data.<br>"
            "• Years <b>beyond your dataset</b> are extrapolated using seasonal patterns + linear trend.<br>"
            "• Confidence bands widen the further out from training data."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        is_future_year = cy_val > max_yr
        badge_label    = "Extrapolated" if is_future_year else "Historical + Model"

        st.markdown(
            f"### 📅 Predictions for {cy_val}"
            f"<span class='custom-year-badge'>{badge_label}</span>",
            unsafe_allow_html=True,
        )

        if is_future_year:
            st.caption(
                f"Year {cy_val} is beyond training data ({min_yr}–{max_yr}). "
                "Values extrapolated via seasonal monthly averages + linear trend from history. "
                "Confidence bands widen the further ahead."
            )
        else:
            n_actual = int(cy_df["Is_Actual"].sum())
            n_total  = len(cy_df)
            st.caption(
                f"{n_actual} of {n_total} month-segment entries use actual recorded data "
                f"({'✅ green' if n_actual == n_total else '🔮 purple = predicted'})."
            )

        # ── KPIs
        ck1, ck2, ck3, ck4 = st.columns(4)
        ck1.metric("📅 Year",              str(cy_val))
        ck2.metric("🔮 Total Revenue",     f"₹{cy_df['Forecast_Sales'].sum()/1e5:,.1f}L")
        ck3.metric("💹 Avg Profit Margin", f"{cy_df['Forecast_Margin_Pct'].mean():.1f}%")
        ck4.metric("📦 Total Units",       f"{int(cy_df['Forecast_Qty'].sum()):,}")
        st.markdown("---")

        # ── Monthly Revenue + Profit bar/line combo
        monthly_cy = (
            cy_df.groupby(["MonthNum","MonthLabel"], as_index=False)
            .agg(Sales=("Forecast_Sales","sum"), Lower=("Forecast_Lower_Sales","sum"),
                 Upper=("Forecast_Upper_Sales","sum"), Net_Profit=("Forecast_Net_Profit","sum"))
            .sort_values("MonthNum")
        )

        fig_cy = go.Figure()
        if is_future_year:
            fig_cy.add_trace(go.Scatter(
                x=pd.concat([monthly_cy["MonthLabel"], monthly_cy["MonthLabel"].iloc[::-1]]),
                y=pd.concat([monthly_cy["Upper"], monthly_cy["Lower"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(176,106,247,0.10)",
                line=dict(color="rgba(255,255,255,0)"), name="Confidence Band", showlegend=True,
            ))
        fig_cy.add_trace(go.Bar(x=monthly_cy["MonthLabel"], y=monthly_cy["Sales"],
                                name="Revenue", marker_color=COLORS["purple"], marker_opacity=0.8))
        fig_cy.add_trace(go.Scatter(x=monthly_cy["MonthLabel"], y=monthly_cy["Net_Profit"],
                                    name="Net Profit", mode="lines+markers",
                                    line=dict(color=COLORS["green"], width=2.5), marker=dict(size=6)))
        fig_cy.update_layout(title=f"Monthly Revenue & Net Profit — {cy_val}", **PLOTLY_LAYOUT,
                              yaxis=dict(tickprefix="₹", tickformat=".2s"), barmode="overlay")
        st.plotly_chart(fig_cy, use_container_width=True)

        # ── Category pie + Region bar
        col_c1, col_c2 = st.columns(2)

        cat_cy = cy_df.groupby("Category", as_index=False)["Forecast_Sales"].sum()
        fig_cy_cat = px.pie(cat_cy, names="Category", values="Forecast_Sales", hole=0.6,
                            color="Category", color_discrete_map=CATEGORY_COLORS,
                            title=f"Revenue by Category — {cy_val}")
        fig_cy_cat.update_traces(textinfo="label+percent")
        fig_cy_cat.update_layout(**PLOTLY_LAYOUT)
        col_c1.plotly_chart(fig_cy_cat, use_container_width=True)

        reg_cy = cy_df.groupby("Region", as_index=False)["Forecast_Sales"].sum().sort_values("Forecast_Sales")
        fig_cy_reg = px.bar(reg_cy, x="Forecast_Sales", y="Region", orientation="h",
                            color="Region", color_discrete_map=REGION_COLORS,
                            title=f"Revenue by Region — {cy_val}")
        fig_cy_reg.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                                  xaxis=dict(tickprefix="₹", tickformat=".2s"))
        col_c2.plotly_chart(fig_cy_reg, use_container_width=True)

        # ── Region × Category heatmap
        heat_cy = (cy_df.groupby(["Region","Category"])["Forecast_Sales"].sum().reset_index()
                   .pivot(index="Region", columns="Category", values="Forecast_Sales").fillna(0))
        fig_cy_heat = px.imshow(heat_cy, color_continuous_scale=["#181b24","#b06af7"],
                                text_auto=".2s", title=f"Revenue Heatmap Region × Category — {cy_val}",
                                aspect="auto")
        fig_cy_heat.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                                   xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_cy_heat, use_container_width=True)

        # ── Margin trend by category
        margin_cy = (cy_df.groupby(["MonthNum","MonthLabel","Category"], as_index=False)
                     ["Forecast_Margin_Pct"].mean().sort_values("MonthNum"))
        fig_cy_marg = px.line(margin_cy, x="MonthLabel", y="Forecast_Margin_Pct",
                              color="Category", color_discrete_map=CATEGORY_COLORS, markers=True,
                              title=f"Monthly Avg Profit Margin % by Category — {cy_val}")
        fig_cy_marg.update_layout(**PLOTLY_LAYOUT, yaxis=dict(title="Margin %"),
                                   xaxis=dict(tickangle=-35))
        st.plotly_chart(fig_cy_marg, use_container_width=True)

        # ── Full detail table
        st.markdown(f"#### Full Month × Segment Table — {cy_val}")
        disp_cy = cy_df[["MonthLabel","Region","Category","Forecast_Sales","Forecast_Margin_Pct",
                          "Forecast_Qty","Forecast_Net_Profit","Forecast_Profit_Class",
                          "Forecast_Lower_Sales","Forecast_Upper_Sales","Is_Actual"]].copy()
        disp_cy = disp_cy.sort_values(["MonthNum" if "MonthNum" in disp_cy.columns else "MonthLabel",
                                        "Region","Category"])

        disp_cy["Forecast_Sales"]       = disp_cy["Forecast_Sales"].apply(lambda v: f"₹{v:,.0f}")
        disp_cy["Forecast_Net_Profit"]  = disp_cy["Forecast_Net_Profit"].apply(lambda v: f"₹{v:,.0f}")
        disp_cy["Forecast_Lower_Sales"] = disp_cy["Forecast_Lower_Sales"].apply(lambda v: f"₹{v:,.0f}")
        disp_cy["Forecast_Upper_Sales"] = disp_cy["Forecast_Upper_Sales"].apply(lambda v: f"₹{v:,.0f}")
        disp_cy["Forecast_Margin_Pct"]  = disp_cy["Forecast_Margin_Pct"].apply(lambda v: f"{v:.1f}%")
        disp_cy["Forecast_Qty"]         = disp_cy["Forecast_Qty"].apply(lambda v: f"{v:,}")
        disp_cy["Is_Actual"]            = disp_cy["Is_Actual"].apply(
            lambda v: "✅ Actual" if v else "🔮 Predicted")

        disp_cy.rename(columns={"MonthLabel":"Month","Forecast_Sales":"Revenue",
                                 "Forecast_Margin_Pct":"Margin","Forecast_Qty":"Units",
                                 "Forecast_Net_Profit":"Net Profit","Forecast_Profit_Class":"Profit Class",
                                 "Forecast_Lower_Sales":"Lower Bound","Forecast_Upper_Sales":"Upper Bound",
                                 "Is_Actual":"Data Source"}, inplace=True)

        def highlight_cy(val):
            if val == "High Profit":
                return "background-color: rgba(61,214,140,0.15); color: #3dd68c; font-weight:600"
            return "background-color: rgba(240,107,107,0.15); color: #f06b6b; font-weight:600"

        def highlight_source(val):
            if val == "✅ Actual":
                return "color: #3dd68c"
            return "color: #b06af7"

        st.dataframe(
            disp_cy.style
                .map(highlight_cy,     subset=["Profit Class"])
                .map(highlight_source, subset=["Data Source"]),
            use_container_width=True, hide_index=True,
        )

        csv_cy = cy_df.drop(columns=["Is_Actual"]).to_csv(index=False).encode("utf-8")
        st.download_button(f"⬇️ Download {cy_val} Predictions CSV", data=csv_cy,
                           file_name=f"predictions_{cy_val}.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────
# TAB 5 · IMPROVEMENTS
# ─────────────────────────────────────────────────────────
with tab_improve:
    st.markdown("### Where You Can Do Better")

    imp_f = improve[improve["Region"].isin(sel_regions) & improve["Category"].isin(sel_cats)].copy()
    col_a, col_b = st.columns(2)
    imp_f["Segment"] = imp_f["Region"] + " · " + imp_f["Category"]

    fig_gap = px.bar(imp_f.sort_values("Margin_Gap_vs_Top_Quartile", ascending=True),
                     x="Margin_Gap_vs_Top_Quartile", y="Segment", orientation="h",
                     color="Margin_Gap_vs_Top_Quartile",
                     color_continuous_scale=["#3dd68c","#f0a842","#f06b6b"],
                     title="Margin Gap vs Top Quartile (pp)")
    fig_gap.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, xaxis=dict(title="Percentage Points"))
    col_a.plotly_chart(fig_gap, use_container_width=True)

    fig_disc = px.bar(imp_f.sort_values("Avg_Discount_Pct", ascending=True),
                      x="Avg_Discount_Pct", y="Segment", orientation="h",
                      color="Avg_Discount_Pct", color_continuous_scale=["#3dd68c","#f0a842","#f06b6b"],
                      title="Avg Discount % by Segment (lower = better)")
    fig_disc.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, xaxis=dict(title="%"))
    col_b.plotly_chart(fig_disc, use_container_width=True)

    fig_inv = px.bar(imp_f.sort_values("Avg_Inventory_Coverage_Days", ascending=False),
                     x="Segment", y="Avg_Inventory_Coverage_Days",
                     color="Avg_Inventory_Coverage_Days",
                     color_continuous_scale=["#3dd68c","#f0a842","#f06b6b"],
                     title="Avg Inventory Coverage Days (target: 30–60)")
    fig_inv.add_hline(y=60, line_dash="dot", line_color=COLORS["red"],   annotation_text="Overstock threshold (60 days)")
    fig_inv.add_hline(y=30, line_dash="dot", line_color=COLORS["amber"], annotation_text="Ideal minimum (30 days)")
    fig_inv.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig_inv, use_container_width=True)

    fig_ful = px.bar(imp_f.sort_values("Fulfillment_Rate_Avg"),
                     x="Fulfillment_Rate_Avg", y="Segment", orientation="h",
                     color="Fulfillment_Rate_Avg", color_continuous_scale=["#f06b6b","#f0a842","#3dd68c"],
                     title="Demand Fulfillment Rate % (higher = better)")
    fig_ful.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, xaxis=dict(title="%"))
    st.plotly_chart(fig_ful, use_container_width=True)

    st.markdown("### 💡 Action Recommendations")
    for _, row in imp_f.sort_values("Margin_Gap_vs_Top_Quartile", ascending=False).iterrows():
        if row["Recommendations"] == "Performance is on-track.":
            icon, border = "✅", "#3dd68c"
        else:
            icon, border = "⚠️", "#f0a842"
        st.markdown(
            f"""<div style="background:#1e2130;border-left:4px solid {border};
                border-radius:0 8px 8px 0;padding:10px 16px;margin-bottom:8px;">
              <strong style="color:#e8eaf0">{icon} {row['Region']} · {row['Category']}</strong>
              <div style="color:#7c8099;font-size:13px;margin-top:4px">
                Margin: <b style="color:#e8eaf0">{row['Avg_Margin_Pct']:.1f}%</b>
                &nbsp;|&nbsp; Gap: <b style="color:#f0a842">{row['Margin_Gap_vs_Top_Quartile']:.1f}pp</b>
                &nbsp;|&nbsp; Discount: <b style="color:#e8eaf0">{row['Avg_Discount_Pct']:.1f}%</b>
              </div>
              <div style="color:#9ca0b0;font-size:12px;margin-top:6px">{row['Recommendations']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────
# TAB 6 · ML MODELS
# ─────────────────────────────────────────────────────────
with tab_models:
    st.markdown("### Trained Model Performance")

    if "Model" in scores_df.columns:
        scores_df = scores_df.set_index("Model")

    model_info = {
        "Sales_Amount":     {"title":"Sales Amount Predictor",  "algo":"Random Forest Regressor",
                             "metric_label":"R²","metric_key":"R2","cv_key":"CV_R2","mae_key":"MAE",
                             "mae_unit":"₹/month","color":COLORS["accent"]},
        "Profit_Margin":    {"title":"Profit Margin Predictor", "algo":"Gradient Boosting Regressor",
                             "metric_label":"R²","metric_key":"R2","cv_key":"CV_R2","mae_key":"MAE",
                             "mae_unit":"% pts","color":COLORS["teal"]},
        "Sell_Quantity":    {"title":"Sell Qty Predictor",      "algo":"Random Forest Regressor",
                             "metric_label":"R²","metric_key":"R2","cv_key":"CV_R2","mae_key":"MAE",
                             "mae_unit":"units","color":COLORS["green"]},
        "Profit_Classifier":{"title":"Profit Classifier",       "algo":"Random Forest Classifier",
                             "metric_label":"Accuracy","metric_key":"Accuracy",
                             "cv_key":None,"mae_key":None,"mae_unit":"","color":COLORS["amber"]},
    }

    cols = st.columns(4)
    for i, (key, info) in enumerate(model_info.items()):
        if key not in scores_df.index:
            cols[i].warning(f"{info['title']}: scores not found")
            continue
        row = scores_df.loc[key]
        primary_val = row.get(info["metric_key"], "N/A")
        with cols[i]:
            st.markdown(
                f"""<div style="background:#181b24;border:1px solid rgba(255,255,255,0.08);
                        border-top:3px solid {info['color']};border-radius:10px;padding:16px;height:180px;">
                  <div style="font-size:11px;color:#7c8099;text-transform:uppercase;letter-spacing:.6px">{info['algo']}</div>
                  <div style="font-size:15px;font-weight:700;color:{info['color']};margin:8px 0 2px">{info['title']}</div>
                  <div style="font-size:28px;font-weight:800;color:#e8eaf0">{float(primary_val):.3f}</div>
                  <div style="font-size:11px;color:#7c8099">{info['metric_label']}</div>
                  {f'<div style="font-size:11px;color:#7c8099;margin-top:4px">CV R²: {float(row.get(info["cv_key"],0)):.3f}</div>' if info.get("cv_key") and info["cv_key"] in row.index else ''}
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.markdown("### Top Feature Importances — Sales Predictor")
    top_fi = fi_df.nlargest(12, "Importance_Sales")
    fig_fi = px.bar(top_fi.sort_values("Importance_Sales"), x="Importance_Sales", y="Feature",
                    orientation="h", color="Importance_Sales",
                    color_continuous_scale=[COLORS["muted"], COLORS["accent"]],
                    title="Feature Importances (Sales Amount Model)")
    fig_fi.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, xaxis=dict(title="Importance Score"))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("### Feature Importance Comparison Across Models")
    top10  = fi_df.nlargest(10, "Importance_Sales")["Feature"].tolist()
    fi_top = (fi_df[fi_df["Feature"].isin(top10)].set_index("Feature")
              [["Importance_Sales","Importance_Margin","Importance_Qty","Importance_Cls"]]
              .rename(columns={"Importance_Sales":"Sales","Importance_Margin":"Margin",
                               "Importance_Qty":"Quantity","Importance_Cls":"Classifier"})
              .reset_index())

    fig_comp = go.Figure()
    for col, color in zip(["Sales","Margin","Quantity","Classifier"],
                           [COLORS["accent"],COLORS["teal"],COLORS["green"],COLORS["amber"]]):
        fig_comp.add_trace(go.Bar(name=col, x=fi_top["Feature"], y=fi_top[col],
                                  marker_color=color, opacity=0.85))
    fig_comp.update_layout(barmode="group", **PLOTLY_LAYOUT,
                            xaxis=dict(tickangle=-30), yaxis=dict(title="Importance"))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("### Full Model Scores")
    st.dataframe(scores_df.reset_index(), use_container_width=True, hide_index=True)