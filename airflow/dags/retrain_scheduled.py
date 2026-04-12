"""
Scheduled solar revenue model retraining DAG.
Runs daily at 2am. Flow: load_data → build_features → train → evaluate → promote.
If evaluate fails (quality gate), promote is skipped.
"""
from datetime import datetime, timedelta
import subprocess
from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def run_script(script_path: str, **context):
    """Generic task runner — executes a Python script and raises on failure."""
    result = subprocess.run(
        ["python", script_path],
        capture_output=True, text=True, cwd="/opt/airflow"
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed:\n{result.stderr}")


def load_new_labels(**context):
    """
    In production: pull yesterday's confirmed revenue from the data warehouse.
    For demo: re-generate data (simulates new data arriving).
    """
    run_script("src/data/generate_data.py", **context)
    print("New labels loaded.")


def build_features(**context):
    run_script("src/data/build_features.py", **context)
    print("Features rebuilt.")


def train_model(**context):
    """Train model and push run_id via xcom for downstream tasks."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.train.train import train_model as _train
    run_id, val_mape = _train()
    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="val_mape", value=val_mape)
    print(f"Training complete. run_id={run_id}, val_mape={val_mape:.2f}%")


def evaluate_and_promote(**context):
    """Quality gate: only promote if metrics pass threshold."""
    import yaml, mlflow
    from mlflow.tracking import MlflowClient

    run_id = context["ti"].xcom_pull(key="run_id")
    val_mape = context["ti"].xcom_pull(key="val_mape")

    with open("/opt/airflow/params.yaml") as f:
        config = yaml.safe_load(f)

    mape_threshold = config["evaluate"]["mape_threshold"]
    if val_mape > mape_threshold:
        raise ValueError(
            f"QUALITY GATE FAILED: val_mape={val_mape:.2f}% > threshold={mape_threshold}%"
        )

    # Promote to Production
    client = MlflowClient()
    model_name = config["serving"]["model_name"]
    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client.transition_model_version_stage(
        model_name, mv.version, "Production", archive_existing_versions=True
    )
    print(f"Promoted {model_name} v{mv.version} → Production. MAPE={val_mape:.2f}%")


with DAG(
    dag_id="solar_model_retrain_scheduled",
    start_date=datetime(2026, 1, 1),
    schedule="0 7 * * *",    # Every day at 7am
    catchup=False,
    default_args=default_args,
    tags=["solar", "mlops", "scheduled"],
) as dag:

    t1 = PythonOperator(task_id="load_new_labels", python_callable=load_new_labels)
    t2 = PythonOperator(task_id="build_features", python_callable=build_features)
    t3 = PythonOperator(task_id="train_model", python_callable=train_model)
    t4 = PythonOperator(task_id="evaluate_and_promote", python_callable=evaluate_and_promote)

    t1 >> t2 >> t3 >> t4