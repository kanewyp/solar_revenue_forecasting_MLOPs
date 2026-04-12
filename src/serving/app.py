"""
FastAPI model serving endpoint.
Loads the Production model from MLflow registry.
Exposes /predict, /health, /metrics
"""

import os
import time
import mlflow.pyfunc 
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from src.serving.metrics import REQUEST_COUNT, REQUEST_LATENCY, PREDICTION_VALUE

# app initialization
app = FastAPI(title="Solar Revenue Forecast API", version='1.0')

# environmental variables
MODEL_NAME = os.getenv("MODEL_NAME", "SolarRevenueModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

# model loading at startup
mlflow.set_tracking_uri(MLFLOW_URI)
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}") 
# (models:/ tells MLFlow to look up the model in the model registry)
# mlflow will queries the registry, finds the artifact path for the Produciton version
# download the weights, and instantiates the LSTMWrapper
print(f"Loaded {MODEL_NAME}/{MODEL_STAGE} from {MLFLOW_URI}")

# Pydantic to enforce that all fields are present, and all types are correct
class PredictionRequest(BaseModel): 
    ghi: float
    temperature: float
    cloud_cover: float
    effective_ghi: float
    capacity_factor: float
    generation_mwh: float
    ppa_price: float
    spot_price: float
    rec_price: float
    curtailment_fraction: float
    sin_doy: float
    cos_doy: float
    month: int
    ghi_lag1: float
    ghi_lag7: float
    generation_lag1: float
    generation_lag7: float
    spot_price_lag1: float
    spot_price_lag7: float
    ghi_roll7: float
    ghi_roll30: float
    spot_price_roll7: float
    spot_ppa_ratio: float

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "stage": MODEL_STAGE}

@app.post("/predict")
def predict(request: PredictionRequest):
    start = time.time()
    try:
        features = pd.DataFrame([request.model_dump()])
        prediction = float(model.predict(features)[0])
        PREDICTION_VALUE.observe(prediction)
        REQUEST_COUNT.labels(status="200").inc()
        return {"predicted_revenue_usd": round(prediction, 2)}
    except Exception as e:
        REQUEST_COUNT.label(status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.observe(time.time() - start)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

