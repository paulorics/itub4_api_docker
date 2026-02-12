# src/inference_api.py  (VERSÃO AJUSTADA PARA LOG-RETURN + Quantile)
# - Mantém compatibilidade com o modelo antigo (retorno %), se você quiser.
# - Adiciona:
#   1) predict_next_log_return_from_features  -> retorna ln(P_{t+1}/P_t)
#   2) predict_next_close_from_features_logreturn_model -> reconstrói preço com exp(log_return)
#
# IMPORTANTE: se você treinou com quantile_loss, precisa carregar o modelo com custom_objects
# (senão keras pode falhar ao carregar). Eu incluí isso em load_bundle.

import os
import json
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras


# -------------------------
# Quantile loss (para load_model)
# -------------------------
def quantile_loss(q: float):
    q = tf.constant(q, dtype=tf.float32)

    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

    return loss


def load_bundle(artifact_dir: str) -> Dict[str, Any]:
    """
    Carrega o bundle de inferência:
      - modelo Keras (.keras)
      - arrays do MinMaxScaler (min_ e scale_) para X e y
      - metadados (lookback, features) a partir de metrics.json

    Espera os arquivos:
      artifact_dir/
        best_model_logret_q60.keras
        x_scaler_min.npy
        x_scaler_scale.npy
        y_scaler_min.npy
        y_scaler_scale.npy
        metrics.json
    """
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"ARTIFACT_DIR não encontrado: {artifact_dir}")

    model_path = os.path.join(artifact_dir, "best_model_logret_q60.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    # Metadados
    metrics_path = os.path.join(artifact_dir, "metrics_logret_q60.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics_logret_q60.json não encontrado: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    lookback = int(meta.get("lookback", 10))
    features = meta.get("features", [])

    if not isinstance(features, list) or len(features) == 0:
        # Ainda funciona sem features, mas você perde validação do K na API
        # K = tamanho de x_min (vamos carregar já já)
        features = None

    # Se você treinou com Quantile Loss, precisamos do custom_objects.
    # Vamos tentar ler um campo do meta; se não existir, usamos 0.60 (padrão do seu projeto).
    q = float(meta.get("quantile_q", 0.60))

    # Carrega modelo (com custom_objects para quantile loss)
    # OBS: mesmo que você não use quantile, isso não atrapalha.
    model = keras.models.load_model(model_path, custom_objects={"loss": quantile_loss(q)})

    # Carrega scaler params
    x_min_path = os.path.join(artifact_dir, "x_scaler_min_logret.npy")
    x_scale_path = os.path.join(artifact_dir, "x_scaler_scale_logret.npy")
    y_min_path = os.path.join(artifact_dir, "y_scaler_min_logret.npy")
    y_scale_path = os.path.join(artifact_dir, "y_scaler_scale_logret.npy")

    for p in [x_min_path, x_scale_path, y_min_path, y_scale_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Arquivo de scaler não encontrado: {p}")

    x_min = np.load(x_min_path).astype(np.float32)       # shape (K,)
    x_scale = np.load(x_scale_path).astype(np.float32)   # shape (K,)
    y_min = np.load(y_min_path).astype(np.float32)       # shape (1,) ou (K_y,)
    y_scale = np.load(y_scale_path).astype(np.float32)   # shape (1,) ou (K_y,)

    if features is None:
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
        "quantile_q": q,
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


# ============================================================
# PREDITORES
# ============================================================

def predict_next_log_return_from_features(
    model,
    X_hist: np.ndarray,
    x_min: np.ndarray,
    x_scale: np.ndarray,
    y_min: np.ndarray,
    y_scale: np.ndarray,
) -> float:
    """
    Prediz o LOG-RETURN do próximo dia:
      y_next_log_return = ln(P_{t+1}/P_t)

    Entrada:
      X_hist: (lookback, K) em escala ORIGINAL (não normalizada)

    Retorna:
      float: log-return previsto (desnormalizado), em escala original (ln).
    """
    X_hist = np.asarray(X_hist, dtype=np.float32)

    if X_hist.ndim != 2:
        raise ValueError("X_hist deve ser 2D no formato (lookback, K).")

    if X_hist.shape[1] != x_min.shape[0]:
        raise ValueError(
            f"Número de features inválido. Esperado K={x_min.shape[0]}, recebido K={X_hist.shape[1]}"
        )

    # 1) Normaliza features
    X_scaled = _minmax_transform(X_hist, x_min=x_min, x_scale=x_scale)  # (lookback, K)

    # 2) Shape para LSTM: (1, lookback, K)
    X_input = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1]).astype(np.float32)

    # 3) Predição em espaço escalado
    y_pred_s = model.predict(X_input, verbose=0).astype(np.float32)  # (1,1)

    # 4) Desnormaliza para log-return (ln) em escala original
    y_pred = _minmax_inverse_transform(y_pred_s, y_min=y_min, y_scale=y_scale)  # (1,1)
    return float(y_pred.reshape(-1)[0])


def predict_next_close_from_features_logreturn_model(
    model,
    X_hist: np.ndarray,
    last_close: float,
    x_min: np.ndarray,
    x_scale: np.ndarray,
    y_min: np.ndarray,
    y_scale: np.ndarray,
) -> float:
    """
    Reconstrói o preço (t+1) a partir do LOG-RETURN previsto:

      log_r = ln(P_{t+1}/P_t)
      => P_{t+1} = P_t * exp(log_r)

    Entrada:
      last_close = Adj Close no tempo t
    """
    log_r = predict_next_log_return_from_features(
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
    return float(last_close * np.exp(log_r))


# ============================================================
# (OPCIONAL) COMPATIBILIDADE COM O MODELO ANTIGO (RETORNO %)
# Se você NÃO usa mais retorno %, pode apagar essa seção.
# ============================================================

def predict_next_return_pct_from_features(
    model,
    X_hist: np.ndarray,
    x_min: np.ndarray,
    x_scale: np.ndarray,
    y_min: np.ndarray,
    y_scale: np.ndarray,
) -> float:
    """
    Prediz o retorno do próximo dia em % (modelo antigo).
    """
    X_hist = np.asarray(X_hist, dtype=np.float32)

    if X_hist.ndim != 2:
        raise ValueError("X_hist deve ser 2D no formato (lookback, K).")

    if X_hist.shape[1] != x_min.shape[0]:
        raise ValueError(
            f"Número de features inválido. Esperado K={x_min.shape[0]}, recebido K={X_hist.shape[1]}"
        )

    X_scaled = _minmax_transform(X_hist, x_min=x_min, x_scale=x_scale)
    X_input = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1]).astype(np.float32)
    y_pred_s = model.predict(X_input, verbose=0).astype(np.float32)
    y_pred = _minmax_inverse_transform(y_pred_s, y_min=y_min, y_scale=y_scale)
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
    Reconstrói o preço (t+1) a partir do retorno % previsto (modelo antigo).
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
