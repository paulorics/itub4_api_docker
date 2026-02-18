# ITUB4.SA – LSTM Predictor API (FastAPI)

## Visão Geral
Este projeto implementa uma API RESTful para **previsão indireta do preço de fechamento (t+1)** da ação **ITUB4.SA**, utilizando um modelo **LSTM** treinado para **prever o retorno percentual do próximo dia**.

Em vez de prever diretamente o preço, o modelo aprende a prever o **retorno (%)**, que é então convertido em preço pela API usando o último preço conhecido:

```
Preço previsto (t+1) = Preço atual (t) × (1 + Retorno previsto / 100)
```

Essa abordagem tende a ser estatisticamente mais estável, pois retornos são séries mais estacionárias do que preços absolutos.

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

> ⚠️ Importante: este projeto assume um **modelo de retorno**.

Também é necessário ter:

- Python 3.9 ou superior
- pip instalado

---

## 2) Rodar localmente (sem Docker – recomendado)

### Por que é necessário criar um ambiente virtual (venv)?

Para garantir que o **Uvicorn (servidor ASGI do FastAPI)** funcione corretamente e evitar conflitos de dependências com outros projetos Python instalados na máquina, é **fortemente recomendado criar e utilizar um ambiente virtual (venv)** antes de instalar os requisitos do projeto.

---

### 🔹 Passo 1 – Criar o ambiente virtual

No diretório raiz do projeto, execute:

```bash
python -m venv venv
```

Isso criará uma pasta chamada `venv/` contendo um ambiente Python isolado exclusivamente para este projeto.

---

### 🔹 Passo 2 – Ativar o ambiente virtual

#### Windows (CMD ou PowerShell)

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

Após ativar, você verá `(venv)` no início da linha do terminal, indicando que o ambiente está ativo.

---

### 🔹 Passo 3 – Instalar as dependências dentro do venv

Com o ambiente virtual ativado, instale as dependências:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Dessa forma, todas as bibliotecas (incluindo FastAPI e Uvicorn) serão instaladas dentro do ambiente isolado.

---

### 🔹 Passo 4 – Executar a API com Uvicorn

Defina a variável de ambiente `ARTIFACT_DIR` e inicie o servidor.

#### Windows

```bash
set ARTIFACT_DIR=artifacts_itub4_return
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Linux / macOS

```bash
export ARTIFACT_DIR=artifacts_itub4_return
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

A API ficará disponível em:

- **Swagger UI (interface interativa):**  
  http://127.0.0.1:8000/docs

- **Healthcheck:**  
  `GET /health`

- **Informações do modelo:**  
  `GET /model_info`

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

O endpoint `/predict` realiza **uma única previsão do próximo dia (t+1)** a partir de **uma única janela temporal já pronta de features**.

> ⚠ **Importante:**  
> O modelo foi treinado com **`lookback = 90`**.  
> Sempre envie exatamente **90 linhas** em `features_history`.  
> O valor `10` mostrado nos exemplos abaixo é apenas ilustrativo para facilitar a compreensão da estrutura.

---

## 📌 Formato esperado da entrada

O corpo da requisição deve conter:

- `features_history`: matriz 2D  
  - **N linhas = LOOKBACK (90 no modelo atual)**  
  - **K colunas = número de features usadas no treinamento**  
  - Deve respeitar exatamente a **ordem das features do treinamento**

- `last_close`: preço de fechamento ajustado (**Adj Close**) do **último dia da janela**

> ❗ Deve ser enviada **apenas uma janela**, não múltiplas janelas.

---

## ✅ Exemplo conceitual (LOOKBACK = 10 – apenas ilustrativo)

Para prever o dia **t+1**, envie as features dos dias:

```
t-9, t-8, t-7, t-6, t-5, t-4, t-3, t-2, t-1, t
```

Cada linha representa **todas as features de um único dia**.

No modelo real (lookback = 90), a lógica é a mesma — apenas com 90 dias consecutivos.

---

## ✅ Exemplo de requisição (JSON)

Supondo as seguintes features utilizadas no treinamento:

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

Exemplo ilustrativo com `LOOKBACK = 10`:

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

> 🔎 No ambiente real, substitua por **90 linhas completas**.

---

## 🔎 Outros endpoints disponíveis

Além do `/predict`, a API também oferece:

### 🔹 `/predict_from_history`

- O usuário envia **dados históricos brutos** (`date`, `adj_close`, `volume`)
- A API executa automaticamente:
  - Engenharia de features
  - Seleção da janela (lookback = 90)
  - Normalização
  - Inferência

Indicado para quem **não deseja montar manualmente as features**.

---

### 🔹 `/predict_auto`

- A API busca automaticamente os dados no Yahoo Finance
- Ticker fixo: **ITUB4.SA**
- Intervalo fixo: **1d**
- O usuário informa apenas `period_days`

Indicado para uso simplificado ou testes rápidos.

---

## 6) Resposta da API

A API retorna:

```json
{
  "ticker": "ITUB4.SA",
  "predicted_return_pct": 0.35,
  "predicted_next_close": 45.34,
  "last_close_used": 45.18,
  "lookback": 90,
  "n_features": 8
}
```

### Campos retornados

- `predicted_return_pct` → retorno previsto para t+1 (%)
- `predicted_next_close` → preço estimado para t+1
- `last_close_used` → preço base utilizado (t)
- `lookback` → tamanho da janela (90)
- `n_features` → número de features do modelo

---

## 7) Observação final

Este modelo foi treinado com:

- **Lookback = 90**
- Features técnicas derivadas de preço e volume
- Normalização via Scikit-Learn (scalers salvos em `.joblib`)
- Arquitetura LSTM para previsão de **retorno percentual**

O projeto foi desenvolvido como parte do **Tech Challenge – FIAP (Fase 4)**, contemplando:

- Deep Learning com LSTM  
- Séries temporais financeiras  
- Pipeline completo de treino → inferência  
- Deploy via API REST  

A abordagem de previsão indireta via retorno foi adotada por sua maior robustez estatística e melhor estabilidade em produção.
