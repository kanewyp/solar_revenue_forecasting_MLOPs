import pandas as pd
import numpy as np
import pytest
import xgboost as xgb
from src.data.generate_data import generate_solar_data
from src.data.build_features import build_features


@pytest.fixture
def sample_features():
    df = generate_solar_data(n_days=200)
    return build_features(df)


def test_model_trains_and_predicts(sample_features):
    import yaml
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    feature_cols = config["train"]["feature_cols"]
    target_col = config["train"]["target_col"]

    X = sample_features[feature_cols]
    y = sample_features[target_col]

    model = xgb.XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X.head(120), y.head(120))
    preds = model.predict(X.tail(30))

    assert len(preds) == 30
    assert not np.any(np.isnan(preds))


def test_model_mape_reasonable(sample_features):
    """Model trained on clean data should have MAPE < 20%."""
    import yaml
    from sklearn.metrics import mean_absolute_percentage_error
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    feature_cols = config["train"]["feature_cols"]
    target_col = config["train"]["target_col"]

    X = sample_features[feature_cols]
    y = sample_features[target_col]

    model = xgb.XGBRegressor(n_estimators=50, random_state=42)
    model.fit(X.head(120), y.head(120))
    preds = model.predict(X.tail(30))
    mape = mean_absolute_percentage_error(y.tail(30), preds) * 100
    assert mape < 20.0, f"MAPE too high: {mape:.2f}%"