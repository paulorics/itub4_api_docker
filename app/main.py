# app/main.py
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference_api import load_bundle, predict_next_return_pct_from_features

# =========================
# Config
# =========================
TICKER_DEFAULT = os.getenv("TICKER", "ITUB4.SA")

# Use a pasta do modelo de RETORNO (para não sobrescrever o modelo de preço direto)
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts_itub4_return")

app = FastAPI(title="ITUB4.SA LSTM Predictor (Return -> Price)", version="1.2.0")


# =========================
# Schemas
# =========================
class PredictFeaturesRequest(BaseModel):
    """
    Low-level endpoint: o usuário fornece as FEATURES prontas.
    """
    features_history: List[List[float]] = Field(
        ...,
        description="Matriz 2D (LOOKBACK x K) na mesma ordem das features usadas no treinamento."
    )
    last_close: float = Field(
        ...,
        description="Adj Close do último dia da janela (P_t). Necessário para reconstruir o preço previsto (t+1)."
    )


class HistoryRow(BaseModel):
    """
    Linha de histórico (mínimo necessário para feature engineering do projeto).
    """
    date: str = Field(..., description="Data no formato YYYY-MM-DD")
    adj_close: float = Field(..., description="Preço de fechamento ajustado (Adj Close)")
    volume: float = Field(..., description="Volume negociado")


class PredictFromHistoryRequest(BaseModel):
    """
    Endpoint principal (alinhado ao enunciado): usuário fornece dados históricos de preços.
    """
    history: List[HistoryRow] = Field(
        ...,
        description="Lista de candles/dados históricos (mínimo: date, adj_close, volume)."
    )
    ticker: Optional[str] = Field(None, description="Opcional. Apenas informativo na resposta.")


class PredictAutoRequest(BaseModel):
    """
    Endpoint de conveniência: a API busca no Yahoo Finance (yfinance).
    """
    ticker: str = Field(TICKER_DEFAULT, description="Ticker no Yahoo Finance (ex.: ITUB4.SA)")
    period_days: int = Field(120, description="Quantos dias para trás buscar (recomendado >= 60; padrão 120)")
    interval: str = Field("1d", description="Intervalo do Yahoo Finance (ex.: 1d, 1h)")


# =========================
# Feature Engineering
# =========================
def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def build_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Espera colunas: ['Adj Close', 'Volume'] e index datetime ordenado.
    Retorna dataframe com features + colunas base.
    """
    feat = df.copy()

    feat["avg_last_21"] = feat["Adj Close"].rolling(window=21).mean()
    feat["avg_last_9"] = feat["Adj Close"].rolling(window=9).mean()

    feat["simple_returns"] = feat["Adj Close"].pct_change() * 100.0
    feat["std_last_20_returns"] = feat["simple_returns"].rolling(window=20).std()
    feat["std_last_5_returns"] = feat["simple_returns"].rolling(window=5).std()
    feat["std_last_5_volume"] = feat["Volume"].rolling(window=5).std()

    feat["rsi"] = rsi_wilder(feat["Adj Close"], period=14)

    # Target pode existir no treino, mas na inferência não é necessário.
    # Mantemos caso você queira inspecionar:
    feat["y_next_return_pct"] = (feat["Adj Close"].shift(-1) / feat["Adj Close"] - 1.0) * 100.0

    return feat.dropna().copy()


def make_window_features(feat: pd.DataFrame, feature_cols: List[str], lookback: int) -> np.ndarray:
    """
    Pega as últimas 'lookback' linhas e retorna matriz (lookback, K) na ordem do treinamento.
    """
    if len(feat) < lookback:
        raise ValueError(f"Histórico insuficiente após feature engineering. Necessário >= {lookback} linhas válidas.")
    X_hist = feat[feature_cols].tail(lookback).values.astype(np.float32)
    return X_hist


# =========================
# Startup: load artifacts
# =========================
@app.on_event("startup")
def startup():
    bundle = load_bundle(ARTIFACT_DIR)
    app.state.model = bundle["model"]
    app.state.x_min = bundle["x_min"]
    app.state.x_scale = bundle["x_scale"]
    app.state.y_min = bundle["y_min"]
    app.state.y_scale = bundle["y_scale"]
    app.state.lookback = int(bundle["lookback"])
    app.state.features = list(bundle["features"])


# =========================
# Basic endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model_info")
def model_info():
    return {
        "ticker_default": TICKER_DEFAULT,
        "artifact_dir": ARTIFACT_DIR,
        "lookback": int(app.state.lookback),
        "features": app.state.features,
        "target": "y_next_return_pct",
        "output": ["predicted_return_pct", "predicted_next_close"],
    }


# =========================
# Endpoint 1: Low-level (features prontas)
# =========================
@app.post("/predict")
def predict_from_features(req: PredictFeaturesRequest):
    lookback = int(app.state.lookback)
    features = app.state.features
    K = len(features)

    if len(req.features_history) != lookback:
        raise HTTPException(status_code=400, detail=f"features_history deve ter {lookback} linhas (lookback).")

    for i, row in enumerate(req.features_history):
        if len(row) != K:
            raise HTTPException(status_code=400, detail=f"Linha {i} deve ter {K} colunas (features).")

    last_close = float(req.last_close)
    if last_close <= 0:
        raise HTTPException(status_code=400, detail="last_close deve ser > 0 (Adj Close do último dia da janela).")

    X_hist = np.array(req.features_history, dtype=np.float32)

    t0 = time.time()
    pred_return_pct = predict_next_return_pct_from_features(
        model=app.state.model,
        X_hist=X_hist,
        x_min=app.state.x_min,
        x_scale=app.state.x_scale,
        y_min=app.state.y_min,
        y_scale=app.state.y_scale,
    )
    predicted_next_close = last_close * (1.0 + (pred_return_pct / 100.0))
    latency_ms = (time.time() - t0) * 1000.0

    return {
        "ticker": TICKER_DEFAULT,
        "predicted_return_pct": float(pred_return_pct),
        "predicted_next_close": float(predicted_next_close),
        "last_close_used": float(last_close),
        "lookback": lookback,
        "n_features": K,
        "latency_ms": latency_ms,
        "mode": "features_direct",
    }


# =========================
# Endpoint 2: Principal (usuário fornece histórico)
# =========================
@app.post("/predict_from_history")
def predict_from_history(req: PredictFromHistoryRequest):
    lookback = int(app.state.lookback)
    feature_cols = app.state.features

    # Constrói DF
    try:
        rows = [{
            "date": r.date,
            "Adj Close": float(r.adj_close),
            "Volume": float(r.volume),
        } for r in req.history]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

        if df.empty:
            raise ValueError("Histórico vazio.")

        # Feature engineering
        feat = build_features_from_df(df)

        # Monta janela lookback e last_close do último dia da janela
        X_hist = make_window_features(feat, feature_cols, lookback)
        last_close = float(feat["Adj Close"].iloc[-1])

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao processar histórico: {e}")

    t0 = time.time()
    pred_return_pct = predict_next_return_pct_from_features(
        model=app.state.model,
        X_hist=X_hist,
        x_min=app.state.x_min,
        x_scale=app.state.x_scale,
        y_min=app.state.y_min,
        y_scale=app.state.y_scale,
    )
    predicted_next_close = last_close * (1.0 + (pred_return_pct / 100.0))
    latency_ms = (time.time() - t0) * 1000.0

    return {
        "ticker": req.ticker or TICKER_DEFAULT,
        "predicted_return_pct": float(pred_return_pct),
        "predicted_next_close": float(predicted_next_close),
        "last_close_used": float(last_close),
        "lookback": lookback,
        "n_features": len(feature_cols),
        "latency_ms": latency_ms,
        "mode": "history_provided",
        "history_rows_received": len(req.history),
        "history_rows_used_after_features": int(len(feat)),
    }


# =========================
# Endpoint 3: Conveniência (busca via yfinance)
# =========================
@app.post("/predict_auto")
def predict_auto(req: PredictAutoRequest):
    lookback = int(app.state.lookback)
    feature_cols = app.state.features

    # Baixa dados do Yahoo
    try:
        end = None
        start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=int(req.period_days))).date().isoformat()

        df = yf.download(
            req.ticker,
            start=start,
            end=end,
            interval=req.interval,
            auto_adjust=False,
            progress=False
        )

        if df is None or df.empty:
            raise ValueError("Nenhum dado retornado do yfinance (df vazio).")

        # Normaliza colunas
        # yfinance vem com colunas padrão: Open High Low Close Adj Close Volume
        needed = ["Adj Close", "Volume"]
        for c in needed:
            if c not in df.columns:
                raise ValueError(f"Coluna obrigatória ausente do yfinance: {c}")

        df = df[needed].dropna().copy()

        # Feature engineering
        feat = build_features_from_df(df)

        # Janela + last_close
        X_hist = make_window_features(feat, feature_cols, lookback)
        last_close = float(feat["Adj Close"].iloc[-1])

    except Exception as e:
        # 502 = erro “upstream” (fonte externa)
        raise HTTPException(status_code=502, detail=f"Falha ao obter/processar dados via yfinance: {e}")

    t0 = time.time()
    pred_return_pct = predict_next_return_pct_from_features(
        model=app.state.model,
        X_hist=X_hist,
        x_min=app.state.x_min,
        x_scale=app.state.x_scale,
        y_min=app.state.y_min,
        y_scale=app.state.y_scale,
    )
    predicted_next_close = last_close * (1.0 + (pred_return_pct / 100.0))
    latency_ms = (time.time() - t0) * 1000.0

    return {
        "ticker": req.ticker,
        "predicted_return_pct": float(pred_return_pct),
        "predicted_next_close": float(predicted_next_close),
        "last_close_used": float(last_close),
        "lookback": lookback,
        "n_features": len(feature_cols),
        "latency_ms": latency_ms,
        "mode": "auto_yfinance",
        "yfinance_start": start,
        "interval": req.interval,
        "rows_downloaded": int(len(df)),
        "rows_used_after_features": int(len(feat)),
    }
