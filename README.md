# sales-forecasting-machine-learning
Machine Learning based Sales Forecasting using Linear Regression, Random Forest and XGBoost with time-series feature engineering and monthly/quarterly predictions.

Sales Forecasting Using Machine Learning

An end-to-end machine learning project for monthly retail sales forecasting using Linear Regression, Random Forest Regressor, and XGBoost. The project includes time-series feature engineering, chronological train/test validation, model comparison, error analysis, and monthly/quarterly forecasting.

> **Portfolio note:** The included dataset is a reproducible synthetic business dataset created for demonstration. This makes the repository safe to publish while keeping the workflow realistic.

Business Problem

Businesses need reliable sales forecasts to plan inventory, staffing, marketing budgets, and cash flow. This project predicts future monthly sales and compares multiple regression algorithms to identify the most accurate approach.

Objectives

• Analyze historical monthly sales patterns and seasonality.
• Engineer lag, rolling, trend, calendar, and business-driver features.
• Train Linear Regression, Random Forest, and XGBoost models.
• Evaluate models using R², RMSE, and MAE.
• Generate 12-month monthly forecasts and quarterly forecasts.
• Translate model results into business planning insights.

Dataset

data/sales_data.csv contains 72 monthly observations from January 2020 to December 2025.

Columns:

• date
• sales
• marketing_spend
• active_customers
• avg_discount_pct
• holiday_season

Feature Engineering

The modelling pipeline creates:

• Year, month and quarter
• Linear time index
• Cyclical month features (sin / cos)
• Sales lag 1, lag 3 and lag 12
• 3-month and 6-month rolling averages
• Marketing spend, customer count, discount rate and holiday indicator

All lag/rolling variables are based on past observations only, reducing look-ahead leakage.

Model Performance

Chronological 80/20 train-test split:

|Model            |R²    |RMSE   |MAE    |
|-----------------|-----:|------:|------:|
|Linear Regression|0.893 |23,899 |18,396 |
|XGBoost          |-2.215|131,212|113,069|
|Random Forest    |-2.891|144,350|126,242|

Best model: Linear Regression
Verified test R²: 0.893

The best model reaches approximately 0.89 R² on the held-out chronological test period. Tree models are also included for comparison; on this trend-heavy synthetic dataset, Linear Regression generalizes better to later periods because tree ensembles do not extrapolate linear growth as naturally.

Business Impact

The forecasting workflow can support:

• Earlier identification of upward/downward sales trends
• Inventory and procurement planning
• Marketing-budget allocation
• Quarterly revenue target setting
• Staffing and operational planning
• Data-driven management decisions

Project Structure

```text
sales_forecasting_ml/
├── data/
│   └── sales_data.csv
├── notebooks/
│   └── sales_forecasting_analysis.ipynb
├── src/
│   ├── eda.py
│   ├── train_model.py
│   └── forecast.py
├── models/
│   └── best_model.joblib
├── outputs/
│   ├── model_metrics.csv
│   ├── test_predictions.csv
│   ├── figures/
│   │   ├── historical_sales_trend.png
│   │   ├── actual_vs_predicted.png
│   │   └── sales_forecast.png
│   └── forecasts/
│       ├── monthly_forecast_12_months.csv
│       └── quarterly_forecast.csv
├── requirements.txt
├── LICENSE
└── README.md
```

How to Run

```bash
git clone <your-repository-url>
cd sales_forecasting_ml
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run exploratory analysis:

```bash
python src/eda.py
```

Train and evaluate models:

```bash
python src/train_model.py
```

Generate future forecasts:

```bash
python src/forecast.py
```

Or open:

```text
notebooks/sales_forecasting_analysis.ipynb
```

Visual Results

Historical Sales Trend

Historical Sales Trend

Actual vs Predicted

Actual vs Predicted

12-Month Forecast

Sales Forecast

Key Technical Skills Demonstrated

Python · Pandas · NumPy · Scikit-learn · XGBoost · Regression · Time-Series Feature Engineering · Model Evaluation · Data Visualization · Forecasting · Business Analytics

Resume Project Description

Sales Forecasting Using Machine Learning

• Developed an end-to-end sales forecasting solution using Linear Regression, Random Forest Regressor and XGBoost.
• Engineered time-series features including lag variables, rolling averages, seasonality and business-driver variables.
• Evaluated models using chronological validation with R², RMSE and MAE and selected the best-performing model.
• Generated monthly and quarterly sales forecasts to support revenue planning, inventory decisions and trend identification.
