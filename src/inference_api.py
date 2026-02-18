import os
import json
from typing import Dict, Any

import numpy as np
import joblib
from tensorflow import keras


def load_bundle(artifact_dir: str) -> Dict[str, Any]:
    """
    Carrega o bundle de inferência:

    Espera os arquivos:
      artifact_dir/
        best_model.keras
        metrics.json
        x_scaler.joblib
        y_scaler.joblib
    """
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"ARTIFACT_DIR não encontrado: {artifact_dir}")

    model_path = os.path.join(artifact_dir, "best_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    x_scaler_path = os.path.join(artifact_dir, "x_scaler.joblib")
    y_scaler_path = os.path.join(artifact_dir, "y_scaler.joblib")
    metrics_path = os.path.join(artifact_dir, "metrics.json")

    for p in [x_scaler_path, y_scaler_path, metrics_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")

    # 1) Modelo
    model = keras.models.load_model(model_path)

    # 2) Scalers (sklearn)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    # 3) Metadados
    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    lookback = int(meta.get("lookback", 90))
    features = meta.get("features", [])

    # fallback de features apenas para não quebrar — mas o ideal é sempre salvar no metrics.json
    if not isinstance(features, list) or len(features) == 0:
        # tenta inferir K pelo scaler
        k = getattr(x_scaler, "n_features_in_", None)
        if k is None:
            raise ValueError("metrics.json sem 'features' e x_scaler sem n_features_in_.")
        features = [f"f{i}" for i in range(int(k))]

    # Validações úteis
    k_scaler = getattr(x_scaler, "n_features_in_", None)
    if k_scaler is not None and k_scaler != len(features):
        raise ValueError(
            f"Inconsistência: x_scaler.n_features_in_={k_scaler} mas len(features)={len(features)}"
        )

    expected_lb = model.input_shape[1]  # normalmente 90
    expected_k = model.input_shape[2]   # K
    if expected_k is not None and expected_k != len(features):
        raise ValueError(
            f"Inconsistência: model espera K={expected_k}, mas metrics.json tem K={len(features)}"
        )
    if expected_lb is not None and expected_lb != lookback:
        raise ValueError(
            f"Inconsistência: model espera lookback={expected_lb}, mas metrics.json tem lookback={lookback}"
        )

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "lookback": lookback,
        "features": features,
        "meta": meta,
    }


def predict_next_return_pct_from_features(
    model,
    X_hist: np.ndarray,
    x_scaler,
    y_scaler,
) -> float:
    """
    Prediz o retorno (%) do próximo dia.

    X_hist: array na escala ORIGINAL, shape (lookback, K)
    """
    X_hist = np.asarray(X_hist, dtype=np.float32)

    if X_hist.ndim != 2:
        raise ValueError("X_hist deve ser 2D no formato (lookback, K).")

    # valida lookback contra o model
    expected_lb = model.input_shape[1]
    expected_k = model.input_shape[2]
    if expected_lb is not None and X_hist.shape[0] != expected_lb:
        raise ValueError(f"Lookback inválido. Esperado {expected_lb}, recebido {X_hist.shape[0]}.")
    if expected_k is not None and X_hist.shape[1] != expected_k:
        raise ValueError(f"Número de features inválido. Esperado {expected_k}, recebido {X_hist.shape[1]}.")

    # 1) Normaliza com o scaler real do sklearn
    X_scaled = x_scaler.transform(X_hist)  # (lookback, K)

    # 2) LSTM input: (1, lookback, K)
    X_input = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1]).astype(np.float32)

    # 3) Predição em espaço escalado
    y_pred_s = model.predict(X_input, verbose=0).astype(np.float32)  # (1,1)

    # 4) Inversão com scaler real do sklearn
    # y_scaler espera 2D: (n, 1)
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1))  # (1,1)
    return float(y_pred.reshape(-1)[0])


def predict_next_close_from_features_return_model(
    model,
    X_hist: np.ndarray,
    last_close: float,
    x_scaler,
    y_scaler,
) -> float:
    r_pct = predict_next_return_pct_from_features(
        model=model,
        X_hist=X_hist,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
    last_close = float(last_close)
    if last_close <= 0:
        raise ValueError("last_close deve ser > 0.")
    return float(last_close * (1.0 + (r_pct / 100.0)))
