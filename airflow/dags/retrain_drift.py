"""
Drift-triggered retraining DAG.
Checks PSI (Population Stability Index) on spot_price feature hourly.
If PSI > 0.2, triggers retraining. Otherwise, skips.

This is exactly the scenario simulated in the data: the price shock at day 500.
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator


PSI_THRESHOLD = 0.2  # Industry standard: >0.2 = significant drift


def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1: no drift. 0.1-0.2: moderate. >0.2: significant drift.
    """
    expected_pcts, bin_edges = np.histogram(expected, bins=n_bins)
    actual_pcts, _ = np.histogram(actual, bins=bin_edges)

    # Add small epsilon to avoid log(0)
    expected_pcts = (expected_pcts / len(expected)) + 1e-6
    actual_pcts = (actual_pcts / len(actual)) + 1e-6

    psi = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return float(psi)


def check_drift(**context) -> str:
    """
    Compares recent spot_price distribution to training baseline.
    Returns task_id to follow: 'train_model' or 'skip_retraining'.
    """
    df = pd.read_csv("/opt/airflow/data/processed/features.csv")

    # Baseline: first 400 rows (pre-drift period)
    baseline = df["spot_price"].values[:400]

    # Recent: last 30 rows (recent production data)
    recent = df["spot_price"].values[-30:]

    psi = calculate_psi(baseline, recent)

    # Push for visibility
    context["ti"].xcom_push(key="psi_score", value=psi)
    print(f"Drift check: PSI={psi:.4f} (threshold={PSI_THRESHOLD})")

    if psi > PSI_THRESHOLD:
        print(f"DRIFT DETECTED (PSI={psi:.4f} > {PSI_THRESHOLD}). Triggering retraining.")
        return "train_model"
    else:
        print(f"No significant drift (PSI={psi:.4f}). Skipping retraining.")
        return "skip_retraining"


def train_model(**context):
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.train.train import train_model as _train
    psi = context["ti"].xcom_pull(key="psi_score")
    run_id, val_mape = _train()
    context["ti"].xcom_push(key="run_id", value=run_id)
    print(f"Retrained due to drift (PSI={psi:.4f}). run_id={run_id}, MAPE={val_mape:.2f}%")


def promote_model(**context):
    import yaml, mlflow
    from mlflow.tracking import MlflowClient
    run_id = context["ti"].xcom_pull(key="run_id")
    with open("/opt/airflow/params.yaml") as f:
        config = yaml.safe_load(f)
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{run_id}/model", config["serving"]["model_name"])
    client.transition_model_version_stage(
        config["serving"]["model_name"], mv.version, "Production",
        archive_existing_versions=True
    )
    print(f"Drift-triggered promotion complete: v{mv.version} → Production")


default_args = {"owner": "mlops-team", "retries": 1, "retry_delay": timedelta(minutes=2)}

with DAG(
    dag_id="solar_model_retrain_drift_triggered",
    start_date=datetime(2026, 1, 1),
    schedule="0 * * * *",   # Check for drift every hour
    catchup=False,
    default_args=default_args,
    tags=["solar", "mlops", "drift"],
) as dag:

    check = BranchPythonOperator(task_id="check_drift", python_callable=check_drift)
    train = PythonOperator(task_id="train_model", python_callable=train_model)
    promote = PythonOperator(task_id="promote_model", python_callable=promote_model)
    skip = EmptyOperator(task_id="skip_retraining")

    check >> [train, skip]
    train >> promote