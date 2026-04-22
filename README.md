# BizzInsight AI — Smart Retail Intelligence

ML-powered analytics dashboard for retail sales, profit prediction, and inventory optimisation.

https://bizz-insight.streamlit.app/

---

## Project Structure

```
your_project_folder/
├── sales_data.csv                ← raw input (required)
├── retail_store_inventory.csv    ← raw input (required)
├── build_dataset.py
├── ml_pipeline.py
├── app.py
├── requirements.txt
└── smart_retail_ml/              ← auto-created by the scripts
    ├── master_dataset.csv
    ├── predictions.csv
    ├── monthly_summary.csv
    ├── yearly_summary.csv
    ├── improvement_report.csv
    ├── feature_importance.csv
    └── model_scores.csv
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the master dataset
```bash
python build_dataset.py
```
This merges `sales_data.csv` + `retail_store_inventory.csv` and engineers all features.
Output: `smart_retail_ml/master_dataset.csv`

### 3. Train models & generate predictions
```bash
python ml_pipeline.py
```
This trains 4 ML models and saves all output CSVs to `smart_retail_ml/`.

### 4. Launch the Streamlit dashboard
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## Raw Data Column Requirements

### sales_data.csv
| Column | Type | Notes |
|---|---|---|
| Sale_Date | date | YYYY-MM-DD |
| Region | string | East / North / South / West |
| Product_Category | string | Clothing / Electronics / Food / Furniture |
| Sales_Amount | float | total ₹ for that transaction |
| Quantity_Sold | int | |
| Unit_Cost | float | cost of goods for that transaction |
| Unit_Price | float | selling price for that transaction |
| Discount | float | 0–1 decimal (e.g. 0.15 = 15%) |
| Product_ID | string / int | used as transaction count |
| Customer_Type | string | New / Returning |
| Sales_Channel | string | Online / Retail |

### retail_store_inventory.csv
| Column | Type | Notes |
|---|---|---|
| Date | date | YYYY-MM-DD |
| Region | string | East / North / South / West |
| Category | string | Clothing / Electronics / **Groceries** / Furniture |
| Inventory Level | float | units on hand |
| Units Sold | int | online units sold |
| Units Ordered | int | |
| Demand Forecast | float | |
| Price | float | |
| Discount | float | 0–1 decimal |
| Weather Condition | string | |
| Holiday/Promotion | int | 0 or 1 |
| Competitor Pricing | float | |
| Seasonality | string | |

> **Note:** The inventory file uses "Groceries" — the pipeline automatically maps this to "Food" to match `sales_data.csv`.

---

## Models

| Model | Algorithm | Target |
|---|---|---|
| Sales Amount Predictor | Random Forest Regressor | Total_Sales_Amount |
| Profit Margin Predictor | Gradient Boosting Regressor | Net_Profit_Margin |
| Sell Quantity Predictor | Random Forest Regressor | Total_Qty_Sold |
| Profit Classifier | Random Forest Classifier | High / Low Profit |

---

## Dashboard Tabs

- **Overview** — KPIs, monthly trend, category & region breakdown, channel split
- **Predictions** — Predicted vs actual scatter, monthly trend, segment table
- **Improvements** — Margin gaps, discount analysis, inventory coverage, action recommendations
- **ML Models** — Model scorecards, feature importances, cross-model comparison
