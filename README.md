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

## 4) Como chamar o endpoint `/predict`

O endpoint `/predict` realiza **uma única previsão** do próximo preço de fechamento (**t+1**) a partir de **uma janela temporal** de dados históricos.

---

### 📌 Formato esperado da entrada (`features_history`)

O campo `features_history` deve ser uma **matriz 2D** contendo:

- **N linhas = LOOKBACK**  
  Cada linha representa **um dia no passado**  
  Exemplo: se `LOOKBACK = 10`, devem ser enviados **os últimos 10 dias**

- **K colunas = número de features**  
  Cada coluna corresponde a uma feature usada no treinamento, **na mesma ordem definida em `FEATURE_COLS`**

👉 **Importante:**  
Você deve enviar **apenas UMA janela temporal**, não múltiplas janelas.

---

### ✅ Exemplo conceitual (LOOKBACK = 10)

Para prever o fechamento do dia **t+1**, envie as features dos dias:
t-9, t-8, t-7, t-6, t-5, t-4, t-3, t-2, t-1, t


Cada linha contém todas as features daquele dia.

---

### ✅ Exemplo de requisição (JSON)

Supondo que o modelo foi treinado com as seguintes features:

```python
FEATURE_COLS = [
  "Adj Close",
    "Volume",
    "simple_returns",
    "std_last_5_returns",
    "std_last_20_returns",
    "avg_last_9",
    "avg_last_21",
    "rsi"
]

E que LOOKBACK = 10, o corpo da requisição deve ser:
{
  "features_history": [
    [1200000, 0.82, 1.15, 0.25, 25.20, 25.00, 55.0],
    [1105000, 0.75, 1.10, 0.10, 25.25, 25.05, 54.3],
    [1302000, 0.90, 1.18, 0.35, 25.30, 25.10, 56.1],
    [1250000, 0.85, 1.14, 0.20, 25.35, 25.15, 55.6],
    [1403000, 0.95, 1.20, 0.40, 25.40, 25.20, 57.0],
    [1358000, 0.92, 1.19, 0.30, 25.45, 25.25, 56.5],
    [1501000, 1.00, 1.25, 0.45, 25.50, 25.30, 58.2],
    [1456000, 0.98, 1.23, 0.38, 25.55, 25.35, 57.8],
    [1604000, 1.05, 1.30, 0.50, 25.60, 25.40, 59.0],
    [1559000, 1.02, 1.28, 0.42, 25.65, 25.45, 58.4]
  ]
}
Observações importantes:

- Cada linha representa um único dia

- A ordem deve ser do dia mais antigo para o mais recente

- Os valores devem estar na escala original (não normalizados)

- A API aplica internamente o MinMaxScaler salvo no treinamento

Veja a ordem correta em **GET /model_info** (campo `features`).
