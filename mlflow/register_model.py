"""
Registers the best MLflow run to the model registry and promotes to Staging.
Run this after training to promote a model through the lifecycle.
"""
import mlflow
from mlflow.tracking import MlflowClient
import argparse

def register_best_run(experiment_name: str, metric: str = "val_mape",
                      mode: str = "min", model_name: str = "SolarRevenueModel"):
    client = MlflowClient()

    # Find the best run in experiment
    runs = mlflow.search_runs(
        experiment_names = [experiment_name],
        filter_string = "",
        order_by = [f"metrics.{metric} {'ASC' if mode == 'min' else 'DESC'}"]
    )
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    best_metric = best_run[f"metrics.{metric}"]
    print(f"Best run: {run_id} | {metric}={best_metric:.4f}")

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
    args = parser.parse_args()

    register_best_run(args.experiment, args.metric, model_name=args.model_name)