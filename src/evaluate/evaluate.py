"""
Model evaluation against hold-out set.
Acts as a quality gate: raises ValueError if metrics fail threshold.
Called by the CI pipeline and the Airflow promotion task.
"""
import json
import sys
import yaml
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def evaluate():
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    eval_cfg = config["evaluate"]
    serve_cfg = config["serving"]

    # Load the current production model
    model_uri = f"models:/{serve_cfg["model_name"]}/{serve_cfg['model_stage']}"
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

    preds = model.predict(pd.DataFrame(X_test, columns=feature_cols))
    val_mape = mean_absolute_percentage_error(y_test, preds) * 100
    val_rmse = np.sqrt(mean_squared_error(y_test, preds))


    print(f"Evaluation — MAPE: {val_mape:.2f}% | RMSE: ${val_rmse:,.0f}")

    if val_mape > eval_cfg["mape_threshold"]:
        raise ValueError(
            f"QUALITY GATE FAILED: MAPE={val_mape:.2f}% > threshold={eval_cfg['mape_threshold']}%"
        )
    if val_rmse > eval_cfg["rmse_threshold"]:
        raise ValueError(
            f"QUALITY GATE FAILED: RMSE=${val_rmse:,.0f} > threshold=${eval_cfg['rmse_threshold']:,.0f}"
        )
    print("Quality gate PASSED.")


if __name__ == "__main__":
    evaluate()