from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "sales_data.csv", parse_dates=["date"])
model = joblib.load(ROOT / "models" / "best_model.joblib")

features = [
    "marketing_spend","active_customers","avg_discount_pct","holiday_season",
    "year","month","quarter","time_index","month_sin","month_cos",
    "sales_lag_1","sales_lag_3","sales_lag_12","rolling_mean_3","rolling_mean_6"
]

history = df.copy()
future_dates = pd.date_range(df["date"].max()+pd.offsets.MonthBegin(1), periods=12, freq="MS")
rows = []
n = len(df)

for i, d in enumerate(future_dates, start=1):
    mo = d.month
    marketing = 42000 + 700*(n+i-1) + 8000*np.sin(2*np.pi*(mo-1)/12)
    customers = 1600 + 24*(n+i-1) + 130*np.sin(2*np.pi*(mo-1)/12)
    discount = 8 + 3*np.sin(2*np.pi*(mo-2)/12)
    holiday = int(mo in [10,11,12])
    s = history["sales"]

    x = pd.DataFrame([{
        "marketing_spend": marketing,
        "active_customers": int(round(customers)),
        "avg_discount_pct": discount,
        "holiday_season": holiday,
        "year": d.year,
        "month": mo,
        "quarter": (mo-1)//3+1,
        "time_index": len(history)-12,
        "month_sin": np.sin(2*np.pi*mo/12),
        "month_cos": np.cos(2*np.pi*mo/12),
        "sales_lag_1": s.iloc[-1],
        "sales_lag_3": s.iloc[-3],
        "sales_lag_12": s.iloc[-12],
        "rolling_mean_3": s.iloc[-3:].mean(),
        "rolling_mean_6": s.iloc[-6:].mean()
    }])[features]

    pred = float(model.predict(x)[0])
    history = pd.concat([history, pd.DataFrame([{
        "date": d, "sales": pred, "marketing_spend": marketing,
        "active_customers": int(round(customers)),
        "avg_discount_pct": discount, "holiday_season": holiday
    }])], ignore_index=True)
    rows.append({"date": d, "forecast_sales": pred})

out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs" / "forecasts" / "monthly_forecast_12_months.csv", index=False)
quarterly = out.assign(quarter=out["date"].dt.to_period("Q").astype(str)).groupby("quarter", as_index=False)["forecast_sales"].sum()
quarterly.to_csv(ROOT / "outputs" / "forecasts" / "quarterly_forecast.csv", index=False)
print(out.to_string(index=False))
print("\nQuarterly Forecast\n", quarterly.to_string(index=False))
