"""
Feature engineering for solar revenue prediction.
Creates lagged features, rolling averages, and cyclical time encodings.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Cyclical time encodings
    doy = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    # Lag features (yesterday, last week)
    for lag in [1, 7, 14]:
        df[f"ghi_lag{lag}"] = df["ghi"].shift(lag)
        df[f"generation_lag{lag}"] = df["generation_mwh"].shift(lag)
        df[f"spot_price_lag{lag}"] = df["spot_price"].shift(lag)

    # Rolling averages 
    for window in [7, 30]:
        df[f"ghi_roll{window}"] = df["ghi"].rolling(window).mean()
        df[f"generation_roll{window}"] = df["generation_mwh"].rolling(window).mean()
        df[f"spot_price_roll{window}"] = df["spot_price"].rolling(window).mean()

    # Price ratio (spot vs PPA - signals market conditions) 
    df["spot_ppa_ratio"] = df["spot_price"] / df["ppa_price"]

    # Drop rows with NaN from lagging
    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/solar_data.csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df_features = build_features(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(args.output, index=False)
    print(f"Features built: {df_features.shape[0]} rows × {df_features.shape[1]} cols → {args.output}")
    