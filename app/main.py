import os
import time
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference_api import load_bundle, predict_next_close_from_features

# Where the trained artifacts live (model + scalers + metadata)
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts_itub4")

app = FastAPI(title="ITUB4.SA LSTM Predictor", version="1.0.0")

class PredictRequest(BaseModel):
    """
    You must provide the last N rows of features (N = lookback used in training).
    These are the same features used in training.
    """
    features_history: List[List[float]] = Field(
        ...,
        description="Lista com N linhas (lookback) e K colunas (features), na mesma ordem do treinamento."
    )

@app.on_event("startup")
def startup():
    bundle = load_bundle(ARTIFACT_DIR)
    app.state.model = bundle["model"]
    app.state.x_min = bundle["x_min"]
    app.state.x_scale = bundle["x_scale"]
    app.state.y_min = bundle["y_min"]
    app.state.y_scale = bundle["y_scale"]
    app.state.lookback = bundle["lookback"]
    app.state.features = bundle["features"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model_info")
def model_info():
    return {
        "ticker": "ITUB4.SA",
        "lookback": app.state.lookback,
        "features": app.state.features,
        "artifact_dir": ARTIFACT_DIR,
    }

@app.post("/predict")
def predict(req: PredictRequest):
    lookback = int(app.state.lookback)
    features = app.state.features
    K = len(features)

    if len(req.features_history) != lookback:
        raise HTTPException(status_code=400, detail=f"features_history deve ter {lookback} linhas (lookback).")

    for i, row in enumerate(req.features_history):
        if len(row) != K:
            raise HTTPException(status_code=400, detail=f"Linha {i} deve ter {K} colunas (features).")

    X_hist = np.array(req.features_history, dtype=np.float32)  # (lookback, K)

    t0 = time.time()
    yhat = predict_next_close_from_features(
        model=app.state.model,
        X_hist=X_hist,
        x_min=app.state.x_min,
        x_scale=app.state.x_scale,
        y_min=app.state.y_min,
        y_scale=app.state.y_scale,
    )
    latency_ms = (time.time() - t0) * 1000.0

    return {
        "ticker": "ITUB4.SA",
        "predicted_next_close": yhat,
        "lookback": lookback,
        "n_features": K,
        "latency_ms": latency_ms,
    }
