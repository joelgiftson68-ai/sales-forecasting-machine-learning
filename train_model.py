import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sales_data.csv"
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "forecasts").mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def make_features(df):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["quarter"] = d["date"].dt.quarter
    d["time_index"] = np.arange(len(d))
    d["month_sin"] = np.sin(2*np.pi*d["month"]/12)
    d["month_cos"] = np.cos(2*np.pi*d["month"]/12)
    d["sales_lag_1"] = d["sales"].shift(1)
    d["sales_lag_3"] = d["sales"].shift(3)
    d["sales_lag_12"] = d["sales"].shift(12)
    d["rolling_mean_3"] = d["sales"].shift(1).rolling(3).mean()
    d["rolling_mean_6"] = d["sales"].shift(1).rolling(6).mean()
    return d.dropna().reset_index(drop=True)

df = pd.read_csv(DATA)
feat = make_features(df)
features = [
    "marketing_spend","active_customers","avg_discount_pct","holiday_season",
    "year","month","quarter","time_index","month_sin","month_cos",
    "sales_lag_1","sales_lag_3","sales_lag_12","rolling_mean_3","rolling_mean_6"
]

split = int(len(feat)*0.8)
train, test = feat.iloc[:split], feat.iloc[split:]
X_train, y_train = train[features], train["sales"]
X_test, y_test = test[features], test["sales"]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=500, max_depth=8, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=3,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        objective="reg:squarederror"
    )
}

rows, predictions = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    rows.append({
        "Model": name,
        "R2": r2_score(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred)**0.5,
        "MAE": mean_absolute_error(y_test, pred)
    })

metrics = pd.DataFrame(rows).sort_values("R2", ascending=False)
metrics.to_csv(OUT / "model_metrics.csv", index=False)
print(metrics.to_string(index=False))

best_name = metrics.iloc[0]["Model"]
best_model = models[best_name]
joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

test_output = test[["date","sales"]].copy()
for name, pred in predictions.items():
    test_output[name.replace(" ","_")] = pred
test_output.to_csv(OUT / "test_predictions.csv", index=False)

plt.figure(figsize=(10,5))
plt.plot(test["date"], y_test, marker="o", label="Actual")
plt.plot(test["date"], predictions[best_name], marker="o", label=f"Predicted - {best_name}")
plt.xticks(rotation=45)
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "figures" / "actual_vs_predicted.png", dpi=160)
