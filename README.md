# solar_revenue_forecasting_MLOPs
Production-grade solar revenue forecasting with automated drift detection &amp; retraining. Stack: XGBoost · MLflow registry · DVC pipeline · Airflow DAGs · FastAPI · Docker · GitHub Actions CI. PSI crosses 0.2 → model retrains automatically. No bad models reach production.

# Full Project Directory Structure

solar-revenue-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
├── airflow/
│   └── dags/
│       ├── retrain_scheduled.py    # Daily retrain DAG
│       └── retrain_drift.py        # Drift-triggered retrain DAG
├── configs/
│   └── train_config.yaml           # All hyperparameters live here (DVC tracks this)
├── data/
│   ├── raw/                        # Raw synthetic solar data (DVC-tracked, not git)
│   └── processed/                  # Feature-engineered data (DVC-tracked)
├── docker/
│   └── serving.Dockerfile          # Inference API container
├── mlflow/
│   └── register_model.py           # Promote model Staging → Production
├── src/
│   ├── data/
│   │   ├── generate_data.py        # Synthetic data generator
│   │   └── build_features.py       # Feature engineering
│   ├── train/
│   │   └── train.py                # XGBoost training + MLflow logging
│   ├── evaluate/
│   │   └── evaluate.py             # Hold-out evaluation + quality gate
│   ├── monitoring/
│   │   └── drift_check.py          # PSI drift detection
│   └── serving/
│       ├── app.py                  # FastAPI inference endpoint
│       └── metrics.py              # Prometheus metrics instrumentation
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── dvc.yaml                        # DVC pipeline definition
├── dvc.lock                        # Exact hashes for reproducibility
├── params.yaml                     # Hyperparameters tracked by DVC
├── docker-compose.yml              # Full stack: API + MLflow + Airflow
├── requirements.txt
└── README.md