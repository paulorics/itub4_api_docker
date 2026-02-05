import json
import os
from typing import Dict, Any

import numpy as np
import tensorflow as tf

def _transform_minmax(X: np.ndarray, min_: np.ndarray, scale_: np.ndarray) -> np.ndarray:
    # sklearn MinMaxScaler: X_scaled = X * scale_ + min_
    return X * scale_ + min_

def _inverse_transform_minmax(Xs: np.ndarray, min_: np.ndarray, scale_: np.ndarray) -> np.ndarray:
    return (Xs - min_) / scale_

def load_bundle(artifact_dir: str) -> Dict[str, Any]:
    """
    Loads:
      - best_model.keras
      - x_scaler_min.npy, x_scaler_scale.npy
      - y_scaler_min.npy, y_scaler_scale.npy
      - metrics.json (for lookback + feature order)
    """
    model_path = os.path.join(artifact_dir, "best_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}. Treine e salve os artifacts primeiro.")

    metrics_path = os.path.join(artifact_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json não encontrado em: {metrics_path}.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    x_min = np.load(os.path.join(artifact_dir, "x_scaler_min.npy"))
    x_scale = np.load(os.path.join(artifact_dir, "x_scaler_scale.npy"))
    y_min = np.load(os.path.join(artifact_dir, "y_scaler_min.npy"))
    y_scale = np.load(os.path.join(artifact_dir, "y_scaler_scale.npy"))

    model = tf.keras.models.load_model(model_path)

    return {
        "model": model,
        "x_min": x_min,
        "x_scale": x_scale,
        "y_min": y_min,
        "y_scale": y_scale,
        "lookback": int(metrics["lookback"]),
        "features": metrics["features"],
        "metrics": metrics,
    }

def predict_next_close_from_features(
    model: tf.keras.Model,
    X_hist: np.ndarray,
    x_min: np.ndarray,
    x_scale: np.ndarray,
    y_min: np.ndarray,
    y_scale: np.ndarray,
) -> float:
    """
    X_hist: (lookback, n_features) in *real* feature space (not scaled)
    returns: predicted next close in real price space
    """
    if X_hist.ndim != 2:
        raise ValueError("X_hist deve ter shape (lookback, n_features).")

    # scale features
    Xs = _transform_minmax(X_hist, x_min, x_scale)  # (lookback, n_features)

    # model expects (1, lookback, n_features)
    X_in = Xs.reshape(1, Xs.shape[0], Xs.shape[1]).astype(np.float32)
    y_pred_s = model.predict(X_in)

    # inverse scale target
    y_pred = _inverse_transform_minmax(y_pred_s, y_min, y_scale)
    return float(y_pred[0, 0])
