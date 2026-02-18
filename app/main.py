# app/main.py
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

from src.inference_api import load_bundle, predict_next_return_pct_from_features
from src.data_loader import fetch_yahoo_prices

# =========================
# Config
# =========================
TICKER_DEFAULT = os.getenv("TICKER", "ITUB4.SA")

# Use a pasta do modelo de RETORNO (Return -> Price)
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts_itub4_return")

app = FastAPI(title="ITUB4.SA LSTM Predictor (Return -> Price)", version="1.3.0")


# =========================
# Schemas
# =========================
class PredictFeaturesRequest(BaseModel):
    """
    Endpoint low-level: usuário fornece FEATURES prontas (LOOKBACK x K) + last_close.
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
    Endpoint principal: usuário fornece dados históricos de preços.
    """
    history: List[HistoryRow] = Field(
        ...,
        description="Lista de dados históricos (mínimo: date, adj_close, volume)."
    )
    ticker: Optional[str] = Field(None, description="Opcional. Apenas informativo na resposta.")


class PredictAutoRequest(BaseModel):
    """
    Endpoint automático.
    
    Observação:
    - O ticker é fixo: ITUB4.SA
    - O intervalo é fixo: 1d
    - A API busca os dados automaticamente via Yahoo Finance.
    """
    period_days: int = Field(
        180,
        description="Quantos dias para trás usar como histórico (padrão 180)"
    )

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

    # Target (não é necessário na inferência, mas mantém para debug/inspeção)
    feat["y_next_return_pct"] = (feat["Adj Close"].shift(-1) / feat["Adj Close"] - 1.0) * 100.0

    return feat.dropna().copy()


def make_window_features(feat: pd.DataFrame, feature_cols: List[str], lookback: int) -> np.ndarray:
    """
    Pega as últimas 'lookback' linhas e retorna matriz (lookback, K) na ordem do treinamento.
    """
    if len(feat) < lookback:
        raise ValueError(
            f"Histórico insuficiente após feature engineering. "
            f"Necessário >= {lookback} linhas válidas. Obtido: {len(feat)}"
        )
    return feat[feature_cols].tail(lookback).values.astype(np.float32)


# =========================
# Startup: load artifacts
# =========================
@app.on_event("startup")
def startup():
    bundle = load_bundle(ARTIFACT_DIR)
    app.state.model = bundle["model"]
    app.state.x_scaler = bundle["x_scaler"]
    app.state.y_scaler = bundle["y_scaler"]
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
@app.post(
    "/predict",
    summary="Previsão a partir de FEATURES prontas (low-level / Return → Price)",
    description="""
Este endpoint é o modo **low-level** da API: você fornece diretamente a matriz de **features já calculadas**
no formato esperado pelo modelo (LSTM), e a API apenas:

1. Valida dimensões (`lookback` x `K`)
2. Normaliza com os scalers salvos (.joblib)
3. Executa o modelo LSTM
4. Retorna o **retorno previsto (%)** e o **próximo preço estimado (t+1)**

### Quando usar
Use este endpoint quando você:
- já tem um pipeline de feature engineering próprio (CSV, banco, ETL, etc.)
- quer total controle sobre as features e sua ordem
- quer evitar qualquer transformação interna de preços/volume

### Entrada esperada
- `features_history`: matriz 2D com shape **(LOOKBACK x K)** na mesma ordem do treinamento
- `last_close`: Adj Close do último dia da janela (P_t). Usado para reconstruir o preço (t+1)

### Regras de validação
- `features_history` deve ter exatamente `lookback` linhas
- cada linha deve ter exatamente `K` colunas (número de features do modelo)
- `last_close` deve ser > 0

### Retorno
- `predicted_return_pct`: retorno previsto para o próximo passo (%)
- `predicted_next_close`: preço estimado do próximo dia (t+1)
- `last_close_used`: preço base usado (t)
- `lookback`: janela utilizada
- `n_features`: número de features (K)
- `latency_ms`: tempo de inferência
""",
    response_description="Resultado da previsão (retorno % e preço t+1) com metadados de execução."
)
def predict_from_features(
    req: PredictFeaturesRequest = Body(
        ...,
        openapi_examples={
            "exemplo_basico": {
                "summary": "Exemplo básico (formato correto)",
                "description": "Exemplo ilustrativo. Ajuste para ter exatamente LOOKBACK linhas e K colunas.",
                "value": {
                    "features_history": [
                        [30.10, 15000000, 0.25, 1.10, 55.2, 29.90, 30.05],
                        [30.15, 14800000, -0.10, 1.05, 54.8, 29.95, 30.07],
                        [30.20, 16000000, 0.30, 1.12, 56.0, 30.00, 30.10]
                    ],
                    "last_close": 30.20
                }
            },
            "exemplo_minimo_ilustrativo": {
                "summary": "Exemplo mínimo (apenas formato)",
                "description": "Apenas para demonstrar o JSON; não necessariamente respeita lookback/K do seu modelo.",
                "value": {
                    "features_history": [
                        [0.0, 0.0],
                        [0.0, 0.0]
                    ],
                    "last_close": 30.0
                }
            }
        },
    )
):
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
        x_scaler=app.state.x_scaler,
        y_scaler=app.state.y_scaler,
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
@app.post(
    "/predict_from_history",
    summary="Previsão a partir de histórico fornecido (Return → Price)",
    description="""
Este endpoint realiza a previsão do **próximo retorno (%)** e do **próximo preço estimado (t+1)**
a partir de um **histórico de preços fornecido pelo usuário**.

### Quando usar
Use este endpoint quando você já possui os dados históricos (ex.: do seu pipeline, banco, CSV)
e quer evitar a coleta automática de dados.

### Entrada esperada
Você deve enviar uma lista `history` com linhas contendo:

- `date` (YYYY-MM-DD)
- `adj_close` (Adj Close)
- `volume` (Volume)

A API irá:
1. Ordenar o histórico por data
2. Executar a engenharia de features (médias móveis, retornos, volatilidade, RSI, etc.)
3. Selecionar as últimas `lookback` linhas válidas após `dropna()`
4. Normalizar usando os scalers salvos (.joblib)
5. Executar o modelo LSTM e retornar:
   - `predicted_return_pct` (retorno previsto em %)
   - `predicted_next_close` (preço estimado t+1 a partir de `last_close_used`)

### Observações importantes
- É necessário enviar **histórico suficiente** para que, após o cálculo das features rolling e o `dropna()`,
  sobrem pelo menos `lookback` linhas válidas.
- Se houver dados insuficientes ou formato inválido, o endpoint retorna **HTTP 400**.

### Retorno
- `predicted_return_pct`: retorno previsto para o próximo passo (%)
- `predicted_next_close`: preço estimado do próximo dia (t+1)
- `last_close_used`: último Adj Close efetivamente usado como base (t)
- `history_rows_received`: quantidade de linhas recebidas
- `history_rows_used_after_features`: linhas restantes após feature engineering (dropna)
""",
    response_description="Resultado da previsão (retorno % e preço t+1) com metadados de execução."
)
def predict_from_history(
    req: PredictFromHistoryRequest = Body(
        ...,
        openapi_examples={
            "exemplo_minimo": {
                "summary": "Exemplo de histórico (formato mínimo)",
                "description": (
                    "Envie um histórico com date/adj_close/volume. "
                    "Na prática, envie linhas suficientes para sobrar >= lookback após as features."
                ),
                "value": {
                    "ticker": "ITUB4.SA",
                    "history": [
                        {"date": "2024-01-02", "adj_close": 30.15, "volume": 15230000},
                        {"date": "2024-01-03", "adj_close": 30.40, "volume": 13120000},
                        {"date": "2024-01-04", "adj_close": 29.98, "volume": 18900000},
                        {"date": "2024-01-05", "adj_close": 30.10, "volume": 14500000},
                        {"date": "2024-01-08", "adj_close": 30.55, "volume": 21000000},
                    ]
                },
            },
            "exemplo_com_mais_dados": {
                "summary": "Exemplo com mais linhas (recomendado)",
                "description": (
                    "Exemplo ilustrativo com mais observações. "
                    "Para lookback=90 e features rolling (ex.: 21), "
                    "use um histórico consideravelmente maior (ex.: 180+ linhas)."
                ),
                "value": {
                    "ticker": "ITUB4.SA",
                    "history": [
                        {"date": "2023-12-18", "adj_close": 29.10, "volume": 18000000},
                        {"date": "2023-12-19", "adj_close": 29.25, "volume": 17500000},
                        {"date": "2023-12-20", "adj_close": 29.05, "volume": 22000000},
                        {"date": "2023-12-21", "adj_close": 29.30, "volume": 16000000},
                        {"date": "2023-12-22", "adj_close": 29.80, "volume": 25000000},
                    ]
                },
            },
        },
    )
):
    lookback = int(app.state.lookback)
    feature_cols = app.state.features

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

        feat = build_features_from_df(df)
        X_hist = make_window_features(feat, feature_cols, lookback)
        last_close = float(feat["Adj Close"].iloc[-1])

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao processar histórico: {type(e).__name__}: {e}")

    t0 = time.time()
    pred_return_pct = predict_next_return_pct_from_features(
        model=app.state.model,
        X_hist=X_hist,
        x_scaler=app.state.x_scaler,
        y_scaler=app.state.y_scaler,
    )

    predicted_next_close = last_close * (1.0 + (pred_return_pct / 100.0))
    latency_ms = (time.time() - t0) * 1000.0

    return {
        "ticker": TICKER_DEFAULT,
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
# Endpoint 3: Conveniência (busca automática via requests data_loader)
# =========================
@app.post(
    "/predict_auto",
    summary="Previsão automática para ITUB4.SA (modelo LSTM de retorno → preço)",
    description="""
Este endpoint realiza **previsão automática do próximo retorno (%) e preço estimado**
para o ativo **ITUB4.SA**, utilizando um modelo LSTM previamente treinado.

### Configuração Fixa

- **Ticker:** ITUB4.SA  
- **Intervalo:** 1d (diário)  
- **Fonte de dados:** Yahoo Finance (via data_loader com múltiplas tentativas)

O usuário informa apenas:

- **period_days:** quantidade de dias anteriores a serem utilizados como base histórica.

---

### Como funciona internamente

1. A API calcula a data inicial com base em `period_days`.
2. Busca um range amplo (10 anos) para garantir estabilidade nas features rolling.
3. Filtra os dados a partir da data calculada.
4. Constrói as features técnicas:
   - Médias móveis (9 e 21 períodos)
   - Retornos percentuais
   - Volatilidade rolling
   - RSI (14)
5. Seleciona as últimas `lookback` observações válidas.
6. Aplica normalização usando os scalers salvos (.joblib).
7. Executa o modelo LSTM.
8. Converte o retorno previsto (%) em preço estimado (t+1).

---

### Retorno

O endpoint retorna:

- `predicted_return_pct` → retorno previsto para o próximo dia (%)
- `predicted_next_close` → preço estimado (t+1)
- `last_close_used` → preço base utilizado (t)
- `lookback` → tamanho da janela usada
- `n_features` → número de features do modelo
- `latency_ms` → tempo de inferência
- `rows_used_after_features` → quantidade de linhas válidas após engenharia de features

---

### Possíveis erros

- **502** → Falha na coleta automática de dados externos
- Histórico insuficiente após engenharia de features
- Dados retornados sem colunas obrigatórias

Este endpoint é recomendado para uso simplificado quando não se deseja enviar
features manualmente.
"""
)
def predict_auto(req: PredictAutoRequest):
    lookback = int(app.state.lookback)
    feature_cols = app.state.features

    # Para garantir estabilidade das features (rolling 21, RSI 14, etc.),
    # buscamos um range maior (ex.: 10y) e filtramos por start.
    # Isso evita depender do parâmetro "range" dinâmico no data_loader,
    # e mantém compatibilidade com a função fetch_yahoo_prices.
    try:
        start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=int(req.period_days))).date().isoformat()

        df = fetch_yahoo_prices(
            symbol="ITUB4.SA",
            start=start,
            end=None,
            range_="10y",          # busca amplo e filtra por start
            interval="1d",
            tries=10,
        )

        if df is None or df.empty:
            raise ValueError("Nenhum dado retornado (df vazio).")

        needed = ["Adj Close", "Volume"]
        for c in needed:
            if c not in df.columns:
                raise ValueError(f"Coluna obrigatória ausente: {c}")

        df = df[needed].dropna().copy()

        feat = build_features_from_df(df)
        X_hist = make_window_features(feat, feature_cols, lookback)
        last_close = float(feat["Adj Close"].iloc[-1])

    except Exception as e:
        # 502 = falha de fonte externa / coleta automática
        raise HTTPException(status_code=502, detail=f"Falha no modo automático: {type(e).__name__}: {e}")

    t0 = time.time()
    pred_return_pct = predict_next_return_pct_from_features(
        model=app.state.model,
        X_hist=X_hist,
        x_scaler=app.state.x_scaler,
        y_scaler=app.state.y_scaler,
    )
    
    predicted_next_close = last_close * (1.0 + (pred_return_pct / 100.0))
    latency_ms = (time.time() - t0) * 1000.0

    return {
        "ticker": TICKER_DEFAULT,
        "predicted_return_pct": float(pred_return_pct),
        "predicted_next_close": float(predicted_next_close),
        "last_close_used": float(last_close),
        "lookback": lookback,
        "n_features": len(feature_cols),
        "latency_ms": latency_ms,
        "mode": "auto_data_loader_requests",
        "interval": "1d",
        "period_days": int(req.period_days),
        "rows_used_after_features": int(len(feat)),
        "start_filter_used": start,
    }
