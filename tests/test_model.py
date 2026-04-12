import pandas as pd
import numpy as np
import pytest
import torch
from src.data.generate_data import generate_solar_data
from src.data.build_features import build_features
from src.train.train import SolarRevenueLSTM, SolarSequenceDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


@pytest.fixture
def sample_features():
    df = generate_solar_data(n_days=200)
    return build_features(df)


def test_dataset_shape(sample_features):
    import yaml
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)["train"]
    feature_cols = cfg["feature_cols"]
    X = sample_features[feature_cols].values.astype(np.float32)
    y = sample_features[cfg["target_col"]].values.astype(np.float32)
    ds = SolarSequenceDataset(X, y, seq_len=cfg["seq_len"])
    x_sample, y_sample = ds[0]
    assert x_sample.shape == (cfg["seq_len"], len(feature_cols))
    assert y_sample.shape == ()


def test_lstm_forward_pass(sample_features):
    import yaml
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)["train"]
    model = SolarRevenueLSTM(
        input_size=len(cfg["feature_cols"]),
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    batch = torch.randn(8, cfg["seq_len"], len(cfg["feature_cols"]))
    output = model(batch)
    assert output.shape == (8,)
    assert not torch.any(torch.isnan(output))


def test_model_trains_and_predicts(sample_features):
    import yaml
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)["train"]
    feature_cols = cfg["feature_cols"]
    X = sample_features[feature_cols].values.astype(np.float32)
    y = sample_features[cfg["target_col"]].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_mean, y_std = y.mean(), y.std()
    y = (y - y_mean) / y_std

    ds = SolarSequenceDataset(X[:120], y[:120], seq_len=cfg["seq_len"])
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = SolarRevenueLSTM(
        input_size=len(feature_cols),
        hidden_size=32,   # smaller for fast test
        num_layers=1,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # 3 epochs is enough to confirm it doesn't crash
    for _ in range(3):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Final predictions should be reasonable
    model.eval()
    X_test = torch.FloatTensor(X[120:150]).unsqueeze(0).expand(1, -1, -1)
    with torch.no_grad():
        pred = model(X_test[:, -cfg["seq_len"]:, :])
    assert not torch.isnan(pred).any()


def test_mape_reasonable(sample_features):
    """After real training, MAPE should be under 20%."""
    import json
    try:
        with open("metrics.json") as f:
            m = json.load(f)
        assert m["val_mape"] < 20.0, f"MAPE too high: {m['val_mape']:.2f}%"
    except FileNotFoundError:
        pytest.skip("metrics.json not found — run train.py first")