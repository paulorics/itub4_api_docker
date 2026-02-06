import os
import json
from typing import Dict, Any, List

import numpy as np
from tensorflow import keras


def load_bundle(artifact_dir: str) -> Dict[str, Any]:
    """
    Carrega o bundle de inferência:
      - modelo Keras (.keras)
      - arrays do MinMaxScaler (min_ e scale_) para X e y
      - metadados (lookback, features) a partir de metrics.json

    Espera os arquivos:
      artifact_dir/
        best_model.keras
        x_scaler_min.npy
        x_scaler_scale.npy
        y_scaler_min.npy
        y_scaler_scale.npy
        metrics.json
    """
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"ARTIFACT_DIR não encontrado: {artifact_dir}")

    model_path = os.path.join(artifact_dir, "best_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    # Carrega modelo
    model = keras.models.load_model(model_path)

    # Carrega scaler params
    x_min_path = os.path.join(artifact_dir, "x_scaler_min.npy")
    x_scale_path = os.path.join(artifact_dir, "x_scaler_scale.npy")
    y_min_path = os.path.join(artifact_dir, "y_scaler_min.npy")
    y_scale_path = os.path.join(artifact_dir, "y_scaler_scale.npy")

    for p in [x_min_path, x_scale_path, y_min_path, y_scale_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Arquivo de scaler não encontrado: {p}")

    x_min = np.load(x_min_path).astype(np.float32)       # shape (K,)
    x_scale = np.load(x_scale_path).astype(np.float32)   # shape (K,)
    y_min = np.load(y_min_path).astype(np.float32)       # shape (1,) ou (K_y,)
    y_scale = np.load(y_scale_path).astype(np.float32)   # shape (1,) ou (K_y,)

    # Metadados
    metrics_path = os.path.join(artifact_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json não encontrado: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    lookback = int(meta.get("lookback", 10))
    features = meta.get("features", [])

    if not isinstance(features, list) or len(features) == 0:
        # Ainda funciona sem features, mas você perde validação do K na API
        features = [f"f{i}" for i in range(int(x_min.shape[0]))]

    bundle = {
        "model": model,
        "x_min": x_min,
        "x_scale": x_scale,
        "y_min": y_min,
        "y_scale": y_scale,
        "lookback": lookback,
        "features": features,
        "meta": meta,
    }
    return bundle


def _minmax_transform(X: np.ndarray, x_min: np.ndarray, x_scale: np.ndarray) -> np.ndarray:
    """
    Aplica a transformação do MinMaxScaler usando os vetores min_ e scale_.
    No sklearn: X_scaled = X * scale_ + min_
    """
    return X * x_scale + x_min


def _minmax_inverse_transform(y_scaled: np.ndarray, y_min: np.ndarray, y_scale: np.ndarray) -> np.ndarray:
    """
    Inverte a transformação do MinMaxScaler usando min_ e scale_:
    No sklearn: X_scaled = X * scale_ + min_
    => X = (X_scaled - min_) / scale_
    """
    return (y_scaled - y_min) / (y_scale + 1e-12)


def predict_next_return_pct_from_features(
    model,
    X_hist: np.ndarray,
    x_min: np.ndarray,
    x_scale: np.ndarray,
    y_min: np.ndarray,
    y_scale: np.ndarray,
) -> float:
    """
    Prediz o retorno do próximo dia em % (y_next_return_pct),
    a partir de uma janela (lookback, K) de features na escala ORIGINAL.

    Retorna:
      float: retorno previsto (%) em escala original (desnormalizado).
    """
    X_hist = np.asarray(X_hist, dtype=np.float32)

    if X_hist.ndim != 2:
        raise ValueError("X_hist deve ser 2D no formato (lookback, K).")

    if X_hist.shape[1] != x_min.shape[0]:
        raise ValueError(
            f"Número de features inválido. Esperado K={x_min.shape[0]}, recebido K={X_hist.shape[1]}"
        )

    # 1) Normaliza features (MinMaxScaler transform)
    X_scaled = _minmax_transform(X_hist, x_min=x_min, x_scale=x_scale)  # (lookback, K)

    # 2) Ajusta shape para LSTM: (1, lookback, K)
    X_input = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1]).astype(np.float32)

    # 3) Predição em espaço escalado
    y_pred_s = model.predict(X_input, verbose=0).astype(np.float32)  # (1,1) típico

    # 4) Desnormaliza para retorno (%) em escala original
    y_pred = _minmax_inverse_transform(y_pred_s, y_min=y_min, y_scale=y_scale)  # (1,1)
    return float(y_pred.reshape(-1)[0])


def predict_next_close_from_features_return_model(
    model,
    X_hist: np.ndarray,
    last_close: float,
    x_min: np.ndarray,
    x_scale: np.ndarray,
    y_min: np.ndarray,
    y_scale: np.ndarray,
) -> float:
    """
    (Opcional) Se o seu modelo prevê retorno (%), esta função reconstrói o preço (t+1)
    usando last_close (Adj Close no tempo t).

    predicted_next_close = last_close * (1 + predicted_return_pct/100)
    """
    r_pct = predict_next_return_pct_from_features(
        model=model,
        X_hist=X_hist,
        x_min=x_min,
        x_scale=x_scale,
        y_min=y_min,
        y_scale=y_scale,
    )
    last_close = float(last_close)
    if last_close <= 0:
        raise ValueError("last_close deve ser > 0.")
    return float(last_close * (1.0 + (r_pct / 100.0)))
