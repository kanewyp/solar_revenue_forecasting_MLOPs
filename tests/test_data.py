import pandas as pd
import pytest
from src.data.generate_data import generate_solar_data
from src.data.build_features import build_features

def test_data_generation_shape():
    df = generate_solar_data(n_days=100, drift_start_day=80)
    assert len(df) == 100
    assert "daily_revenue_usd" in df.columns
    assert "ghi" in df.columns

def test_data_no_negative_revenue():
    df = generate_solar_data(n_days=200)
    assert (df["daily_revenue_usd"] >= 0).all()

def test_feature_engineering():
    df = generate_solar_data(n_days=100)
    df_features = build_features(df)
    assert "sin_doy" in df_features.columns
    assert "spot_ppa_ratio" in df_features.columns
    # Rolling features need warmup rows, so fewer rows expected
    assert len(df_features) < len(df)
    assert not df_features.isnull().any().any()

def test_drift_period_labeling():
    df = generate_solar_data(n_days=200, drift_start_day=150)
    assert df["is_drift_period"].sum() == 50