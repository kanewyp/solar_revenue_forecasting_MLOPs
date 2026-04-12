"""
Registers the best MLflow run to the model registry and promotes to Staging.
Run this after training to promote a model through the lifecycle.
"""
import os
import mlflow
from mlflow.tracking import MlflowClient
import argparse


def _has_logged_model_artifact(client: MlflowClient, run_id: str) -> bool:
    """Return True only when run has a model artifact with MLmodel metadata file."""
    try:
        model_items = client.list_artifacts(run_id, path="model")
    except Exception:
        return False
    return any(item.path.endswith("MLmodel") for item in model_items)

def register_best_run(
    experiment_name: str,
    metric: str = "val_mape",
    mode: str = "min",
    model_name: str = "SolarRevenueModel",
    selection: str = "latest",
):
    client = MlflowClient()

    # Find the best run in experiment
    if selection == "latest":
        order_by = ["attributes.start_time DESC"]
    else:
        order_by = [f"metrics.{metric} {'ASC' if mode == 'min' else 'DESC'}"]

    runs = mlflow.search_runs(
        experiment_names = [experiment_name],
        filter_string = "",
        order_by = order_by,
    )
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = None
    for _, candidate in runs.iterrows():
        candidate_run_id = candidate["run_id"]
        if _has_logged_model_artifact(client, candidate_run_id):
            best_run = candidate
            break

    if best_run is None:
        raise ValueError(
            "No runs contain a logged model artifact at 'model/MLmodel'. "
            "Run training first (with MLFLOW_TRACKING_URI pointing to your server)."
        )

    run_id = best_run["run_id"]
    best_metric = best_run[f"metrics.{metric}"]
    print(f"Selected run: {run_id} | {metric}={best_metric:.4f} | selection={selection}")

    # Register the model
    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    print(f"Registered as {model_name} v{mv.version}")

    # Transition to Staging
    client.transition_model_version_stage(model_name, mv.version, "Staging")
    print(f"Transitioned {model_name} v{mv.version} → Staging")

    # After validation (for demo: auto-promote to Production)
    client.transition_model_version_stage(
        model_name, mv.version, "Production", archive_existing_versions=True
    )
    print(f"Promoted {model_name} v{mv.version} → Production")
    return run_id, mv.version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="solar-revenue-forecasting")
    parser.add_argument("--metric", default="val_mape")
    parser.add_argument("--model-name", default="SolarRevenueModel")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    parser.add_argument("--selection", choices=["latest", "best_metric"], default="latest")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    register_best_run(
        args.experiment,
        args.metric,
        model_name=args.model_name,
        selection=args.selection,
    )