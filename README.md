# ITUB4.SA – LSTM Predictor API (FastAPI + Docker)

## 1) Pré-requisito
Você precisa ter treinado o modelo e gerado a pasta **artifacts_itub4/** contendo:
- best_model.keras
- x_scaler_min.npy / x_scaler_scale.npy
- y_scaler_min.npy / y_scaler_scale.npy
- metrics.json

> Se você usou o notebook de treino, ele já salva isso automaticamente.

## 2) Rodar localmente (sem Docker)
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Acesse:
- GET /health
- GET /model_info
- POST /predict

## 3) Rodar com Docker
```bash
docker build -t itub4-lstm-api .
docker run -p 8000:8000 itub4-lstm-api
```

## 4) Como chamar o endpoint /predict
O endpoint espera `features_history` com:
- **N linhas** = lookback (ex.: 60)
- **K colunas** = número de features (mesma ordem do treinamento)

Exemplo (placeholder):
```json
{
  "features_history": [
    [0.1, 0.2, 0.3, 0.4, 25.1, 25.0, 55.0],
    ...
  ]
}
```

Veja a ordem correta em **GET /model_info** (campo `features`).
