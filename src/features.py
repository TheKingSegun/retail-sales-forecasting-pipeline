"""
Feature Engineering for Retail Sales Forecasting
Lag features, rolling statistics, calendar effects, and Nigerian market specifics.
"""
import pandas as pd
import numpy as np
from typing import List

NIGERIAN_HOLIDAYS = [
    "2024-01-01", "2024-04-01", "2024-04-29", "2024-06-12",
    "2024-10-01", "2024-12-25", "2024-12-26",
]

def add_lag_features(df, target_col="sales", lags=[7, 14, 21, 28, 35]):
    df = df.sort_values("date").copy()
    for lag in lags:
        df[f"lag_{lag}d"] = df.groupby("store_id")[target_col].shift(lag)
    return df

def add_rolling_features(df, target_col="sales", windows=[7, 14, 28]):
    for w in windows:
        grp = df.groupby("store_id")[target_col]
        df[f"rolling_{w}d_mean"] = grp.transform(lambda x: x.shift(1).rolling(w).mean())
        df[f"rolling_{w}d_std"]  = grp.transform(lambda x: x.shift(1).rolling(w).std())
        df[f"rolling_{w}d_max"]  = grp.transform(lambda x: x.shift(1).rolling(w).max())
    return df

def add_calendar_features(df, date_col="date"):
    df[date_col] = pd.to_datetime(df[date_col])
    df["day_of_week"]         = df[date_col].dt.dayofweek
    df["day_of_month"]        = df[date_col].dt.day
    df["week_of_year"]        = df[date_col].dt.isocalendar().week.astype(int)
    df["month"]               = df[date_col].dt.month
    df["quarter"]             = df[date_col].dt.quarter
    df["is_weekend"]          = (df["day_of_week"] >= 4).astype(int)
    df["is_month_end"]        = df[date_col].dt.is_month_end.astype(int)
    df["is_month_start"]      = df[date_col].dt.is_month_start.astype(int)
    df["is_salary_week"]      = (df["day_of_month"] >= 25).astype(int)
    df["is_rainy_season"]     = df["month"].between(4, 10).astype(int)
    holidays = pd.to_datetime(NIGERIAN_HOLIDAYS)
    df["is_nigerian_holiday"] = df[date_col].isin(holidays).astype(int)
    return df

def build_feature_matrix(df, target_col="sales"):
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = add_calendar_features(df)
    return df.dropna()

def generate_sample_retail_data(n_stores=10, n_days=730):
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    rows = []
    for store_id in range(1, n_stores + 1):
        base = np.random.uniform(80000, 500000)
        trend = np.linspace(1.0, 1.15, n_days)
        seas = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        weekly = np.array([0.85,0.88,0.92,0.95,1.05,1.20,1.15])[np.arange(n_days) % 7]
        noise = np.random.normal(1.0, 0.08, n_days)
        sales = base * trend * seas * weekly * noise
        for i, date in enumerate(dates):
            rows.append({
                "date": date,
                "store_id": f"STORE_{store_id:03d}",
                "sales": max(0, round(sales[i], 2)),
                "customers": int(sales[i] / np.random.uniform(3000, 8000)),
                "is_promo": int(np.random.random() < 0.15),
                "open": 1 if date.weekday() < 6 else int(np.random.random() < 0.7),
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_sample_retail_data(n_stores=3, n_days=365)
    df = build_feature_matrix(df)
    print(f"Feature matrix shape: {df.shape}")
    print(df.dtypes)
