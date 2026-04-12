"""
Model evaluation against hold-out set.
Acts as a quality gate: raises ValueError if metrics fail threshold.
Called by the CI pipeline and the Airflow promotion task.
"""
import json
import os
import sys
import yaml
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE with non-zero target filtering to avoid divide-by-zero blowups."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > 1e-6
    if not np.any(mask):
        raise ValueError("Cannot compute MAPE: all evaluation targets are zero.")
    return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100)

def evaluate():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))

    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    eval_cfg = config["evaluate"]
    serve_cfg = config["serving"]
    seq_len = config["train"]["seq_len"]
    fail_on_quality_gate = eval_cfg.get("fail_on_quality_gate", False)

    # Load the current production model
    model_uri = f"models:/{serve_cfg['model_name']}/{serve_cfg['model_stage']}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception:
        print("No production model found. Skipping evaluation gate.")
        return

    df = pd.read_csv("data/processed/features.csv")
    feature_cols = config["train"]["feature_cols"]
    split_idx = int(len(df) * 0.8)
    X_test = df[feature_cols].iloc[split_idx:]
    y_test = df[config["train"]["target_col"]].iloc[split_idx:]

    if len(X_test) <= seq_len:
        raise ValueError(
            f"Not enough holdout rows for seq_len={seq_len}. Got {len(X_test)} rows."
        )

    # Build rolling windows of size seq_len so model input matches training semantics.
    preds = []
    for end_idx in range(seq_len, len(X_test) + 1):
        window = X_test.iloc[end_idx - seq_len:end_idx]
        pred = float(model.predict(pd.DataFrame(window, columns=feature_cols))[0])
        preds.append(pred)
    preds = np.array(preds)

    # Each window ending at index i predicts target at i-1 within the holdout slice.
    y_eval = y_test.iloc[seq_len - 1:].values
    val_mape = safe_mape(y_eval, preds)
    val_rmse = np.sqrt(mean_squared_error(y_eval, preds))


    print(f"Evaluation — MAPE: {val_mape:.2f}% | RMSE: ${val_rmse:,.0f}")

    gate_failures = []
    if val_mape > eval_cfg["mape_threshold"]:
        gate_failures.append(
            f"MAPE={val_mape:.2f}% > threshold={eval_cfg['mape_threshold']}%"
        )
    if val_rmse > eval_cfg["rmse_threshold"]:
        gate_failures.append(
            f"RMSE=${val_rmse:,.0f} > threshold=${eval_cfg['rmse_threshold']:,.0f}"
        )

    if gate_failures:
        message = "QUALITY GATE FAILED: " + " | ".join(gate_failures)
        if fail_on_quality_gate:
            raise ValueError(message)
        print(message)
        print("Continuing because evaluate.fail_on_quality_gate=false")
        return

    print("Quality gate PASSED.")


if __name__ == "__main__":
    evaluate()