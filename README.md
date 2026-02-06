# ITUB4.SA – LSTM Predictor API (FastAPI)

## Visão Geral
Este projeto implementa uma API RESTful para **previsão indireta do preço de fechamento (t+1)** da ação **ITUB4.SA**, utilizando um modelo **LSTM** treinado para **prever o retorno percentual do próximo dia**.

Em vez de prever diretamente o preço, o modelo aprende a prever o **retorno (%)**, que é então convertido em preço pela API usando o último preço conhecido:

```
Preço previsto (t+1) = Preço atual (t) × (1 + Retorno previsto / 100)
```

Essa abordagem costuma ser mais estável estatisticamente, pois retornos são séries mais estacionárias do que preços absolutos.

---

## 1) Pré-requisitos
Antes de rodar a API, é necessário ter treinado o modelo e gerado a pasta de artefatos do **modelo de retorno**:

```
artifacts_itub4_return/
  ├── best_model.keras
  ├── x_scaler_min.npy
  ├── x_scaler_scale.npy
  ├── y_scaler_min.npy
  ├── y_scaler_scale.npy
  └── metrics.json
```

> ⚠️ Importante: este projeto assume **modelo de retorno**. Caso você também tenha um modelo de preço direto, mantenha os artefatos em pastas separadas.

---

## 2) Rodar localmente (sem Docker – recomendado)

Ative o ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a API apontando para o arquivo principal (`app/main.py`):

```bash
set ARTIFACT_DIR=artifacts_itub4_return
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

A API ficará disponível em:

- Swagger UI (interface de testes): http://127.0.0.1:8000/docs
- Healthcheck: `GET /health`
- Informações do modelo: `GET /model_info`

> ℹ️ Acessar `http://127.0.0.1:8000/` pode retornar **404 Not Found**. Isso é esperado, pois a API não define rota raiz.

---

## 3) Rodar com Docker (opcional)

O Docker é recomendado apenas para deploy ou portabilidade.

```bash
docker build -t itub4-lstm-api .
docker run -p 8000:8000 -e ARTIFACT_DIR=artifacts_itub4_return itub4-lstm-api
```

---

## 4) Como funciona a previsão indireta

1. A API recebe uma **janela temporal** de features históricas (`features_history`)
2. O modelo LSTM prevê o **retorno percentual do próximo dia** (`y_next_return_pct`)
3. A API reconstrói o preço usando o último `Adj Close` informado pelo usuário

Isso permite que o endpoint retorne:

- Retorno previsto (%)
- Preço de fechamento previsto (t+1)

---

## 5) Como chamar o endpoint `/predict`

O endpoint `/predict` realiza **uma única previsão** do próximo dia (**t+1**) a partir de **uma única janela temporal**.

### 📌 Formato esperado da entrada

O corpo da requisição deve conter:

- `features_history`: matriz 2D
  - **N linhas = LOOKBACK** (ex.: 10)
  - **K colunas = número de features** (mesma ordem do treinamento)
- `last_close`: preço de fechamento ajustado (Adj Close) do **último dia da janela**

> ❗ Você deve enviar **apenas uma janela**, não múltiplas janelas.

---

### ✅ Exemplo conceitual (LOOKBACK = 10)

Para prever o dia **t+1**, envie as features dos dias:

```
t-9, t-8, t-7, t-6, t-5, t-4, t-3, t-2, t-1, t
```

Cada linha contém **todas as features daquele dia**.

---

### ✅ Exemplo de requisição (JSON)

Supondo as seguintes features usadas no treinamento:

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
```

Com `LOOKBACK = 10`, o corpo da requisição será:

```json
{
  "last_close": 45.18,
  "features_history": [
    [45.00, 1200000, 0.82, 1.15, 0.25, 25.20, 25.00, 55.0],
    [45.05, 1105000, 0.75, 1.10, 0.10, 25.25, 25.05, 54.3],
    [45.10, 1302000, 0.90, 1.18, 0.35, 25.30, 25.10, 56.1],
    [45.15, 1250000, 0.85, 1.14, 0.20, 25.35, 25.15, 55.6],
    [45.20, 1403000, 0.95, 1.20, 0.40, 25.40, 25.20, 57.0],
    [45.25, 1358000, 0.92, 1.19, 0.30, 25.45, 25.25, 56.5],
    [45.30, 1501000, 1.00, 1.25, 0.45, 25.50, 25.30, 58.2],
    [45.35, 1456000, 0.98, 1.23, 0.38, 25.55, 25.35, 57.8],
    [45.40, 1604000, 1.05, 1.30, 0.50, 25.60, 25.40, 59.0],
    [45.18, 1559000, 1.02, 1.28, 0.42, 25.65, 25.45, 58.4]
  ]
}
```

### Observações importantes

- Cada linha representa **um único dia**
- A ordem deve ser do dia **mais antigo → mais recente**
- Os valores devem estar na **escala original** (não normalizados)
- A normalização é aplicada internamente pela API
- Consulte `GET /model_info` para confirmar `lookback` e ordem das features

---

## 6) Resposta da API

A API retorna tanto o **retorno previsto** quanto o **preço reconstruído**:

```json
{
  "ticker": "ITUB4.SA",
  "predicted_return_pct": 0.35,
  "predicted_next_close": 45.34,
  "last_close_used": 45.18,
  "lookback": 10,
  "n_features": 8
}
```

---

## 7) Observação final

Este projeto foi desenvolvido como parte do **Tech Challenge – FIAP (Fase 4)**, contemplando:

- Deep Learning com LSTM
- Séries temporais financeiras
- Pipeline de treino → inferência
- Deploy via API REST

A abordagem de previsão indireta via retorno foi adotada por sua maior robustez estatística e melhor comportamento em produção.

