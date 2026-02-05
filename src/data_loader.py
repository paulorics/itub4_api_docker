import time, random
import requests
import pandas as pd

def fetch_yahoo_prices(
    symbol: str,
    start: str = None,
    end: str = None,
    range_: str = "5y",
    interval: str = "1d",
    tries: int = 10,
) -> pd.DataFrame:

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        "Referer": f"https://finance.yahoo.com/quote/{symbol}",
        "Connection": "keep-alive",
    })

    # Warm-up cookies
    s.get("https://finance.yahoo.com/", timeout=20)
    s.get(f"https://finance.yahoo.com/quote/{symbol}", timeout=20)

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": range_, "interval": interval}

    last_error = None

    for i in range(tries):
        try:
            r = s.get(url, params=params, timeout=20)
            text = r.text.strip()

            if not text or not text.startswith("{"):
                raise ValueError("Resposta não JSON")

            j = r.json()
            result = j["chart"]["result"][0]

            ts = result["timestamp"]
            q = result["indicators"]["quote"][0]
            ac = result["indicators"].get("adjclose", [None])[0]
            adj = ac["adjclose"] if ac else q["close"]

            df = pd.DataFrame({
                "Open": q["open"],
                "High": q["high"],
                "Low": q["low"],
                "Close": q["close"],
                "Adj Close": adj,
                "Volume": q["volume"],
            }, index=pd.to_datetime(ts, unit="s"))

            df.index.name = "Date"
            df = df.dropna()

            if start:
                df = df[df.index >= pd.to_datetime(start)]
            if end:
                df = df[df.index <= pd.to_datetime(end)]

            if df.empty:
                raise ValueError("DataFrame vazio após filtros")

            return df

        except Exception as e:
            last_error = str(e)
            wait = min(120, (2 ** i) + random.uniform(0.5, 2.0))
            time.sleep(wait)

    raise RuntimeError(f"Falha ao coletar {symbol}. Último erro: {last_error}")
