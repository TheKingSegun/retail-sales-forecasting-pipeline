"""
Retail Sales Forecasting — Multi-model training with MLflow tracking
Compares LightGBM, Prophet, and SARIMA on Lagos FMCG retail data.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW = True
except ImportError:
    MLFLOW = False

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from features import generate_sample_retail_data, build_feature_matrix

LAG_COLS = [f"lag_{d}d" for d in [7, 14, 21, 28, 35]]
CALENDAR_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter",
    "is_weekend", "is_month_end", "is_salary_week", "is_rainy_season",
    "is_nigerian_holiday", "is_promo", "open",
]


def evaluate(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
    return {"mape": round(mape, 2), "rmse": round(rmse, 2), "mae": round(mae, 2)}


def train_lightgbm(X_train, y_train, X_val, y_val):
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        metrics = evaluate(y_val, model.predict(X_val))
        print(f"  LightGBM  MAPE={metrics['mape']:.1f}%  RMSE={metrics['rmse']:,.0f}")
        return model, metrics
    except ImportError:
        print("  LightGBM not available")
        return None, {}


def train_prophet(store_df: pd.DataFrame):
    try:
        from prophet import Prophet
        ts = store_df[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
        split = int(len(ts) * 0.8)
        m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True,
                    weekly_seasonality=True, changepoint_prior_scale=0.05)
        m.fit(ts.head(split))
        future  = m.make_future_dataframe(periods=len(ts) - split)
        forecast = m.predict(future)
        val_pred   = forecast.tail(len(ts) - split)["yhat"].values
        val_actual = ts.tail(len(ts) - split)["y"].values
        metrics = evaluate(val_actual, val_pred)
        print(f"  Prophet   MAPE={metrics['mape']:.1f}%  RMSE={metrics['rmse']:,.0f}")
        return m, metrics
    except ImportError:
        print("  Prophet not available")
        return None, {}


def run_training():
    print("Generating retail data...")
    raw = generate_sample_retail_data(n_stores=5, n_days=365)
    feat = build_feature_matrix(raw)

    feature_cols = [c for c in feat.columns
                    if c not in ["date", "store_id", "sales", "customers"]]

    if MLFLOW:
        mlflow.set_experiment("retail-forecasting-nigeria")

    all_results = {}
    for store in feat["store_id"].unique()[:3]:
        print(f"\n--- {store} ---")
        sd = feat[feat["store_id"] == store].sort_values("date")
        split = int(len(sd) * 0.8)
        X_train, y_train = sd.head(split)[feature_cols], sd.head(split)["sales"]
        X_val,   y_val   = sd.tail(len(sd)-split)[feature_cols], sd.tail(len(sd)-split)["sales"]

        lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val)

        store_raw = raw[raw["store_id"] == store].reset_index(drop=True)
        prophet_model, prophet_metrics = train_prophet(store_raw)

        if MLFLOW and lgb_model:
            with mlflow.start_run(run_name=f"lgbm_{store}"):
                mlflow.log_params({"store": store, "model": "lightgbm"})
                mlflow.log_metrics(lgb_metrics)

        all_results[store] = {"lightgbm": lgb_metrics, "prophet": prophet_metrics}

    print("\n=== Results Summary ===")
    for store, res in all_results.items():
        lgb = res["lightgbm"]
        print(f"  {store}: LightGBM MAPE={lgb.get('mape','N/A')}%")


if __name__ == "__main__":
    run_training()
