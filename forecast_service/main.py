# forecast_service/main.py
import os
import re

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import Kronos, KronosTokenizer, KronosPredictor

from .attention import predict_with_attention

app = FastAPI()

default_origins = "http://localhost:5173,http://127.0.0.1:5173"
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
# device="cpu": Kronos' bit-quantization ops crash on Apple MPS
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

SYMBOL_RE = re.compile(r"^[A-Za-z0-9.&-]{1,20}$")


@app.post("/forecast")
def forecast(payload: dict):
    df = pd.DataFrame(payload["candles"])  # open, high, low, close, volume
    x_ts = pd.to_datetime(pd.Series(payload["timestamps"]))
    y_ts = pd.to_datetime(pd.Series(payload["future_timestamps"]))
    pred_df = predictor.predict(
        df=df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=payload.get("pred_len", 30), T=1.0, top_p=0.9, sample_count=1
    )
    return pred_df.reset_index().to_dict(orient="records")


@app.get("/api/predict")
def predict_symbol(symbol: str, history_days: int = 180, pred_len: int = 10):
    symbol = symbol.strip().upper()
    if not SYMBOL_RE.match(symbol):
        raise HTTPException(status_code=400, detail="Invalid NSE symbol.")
    ticker = symbol if "." in symbol else f"{symbol}.NS"

    hist = yf.Ticker(ticker).history(period="2y", interval="1d")
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No NSE data found for '{symbol}'.")

    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    hist = hist.tail(min(history_days, 512))

    candles_df = hist[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    x_ts = hist["Date"]

    last_date = hist["Date"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)

    pred_df, attn_weights, historical_reliance = predict_with_attention(
        predictor, df=candles_df, x_timestamp=x_ts, y_timestamp=pd.Series(future_dates),
        pred_len=pred_len, T=1.0, top_p=0.9
    )

    attn_dates = hist["Date"].dt.strftime("%Y-%m-%d").tolist()
    attention = [{"date": d, "weight": float(w)} for d, w in zip(attn_dates, attn_weights)]

    recent_k = min(10, len(attention))
    recent_share = float(sum(a["weight"] for a in attention[-recent_k:]))
    top = max(attention, key=lambda a: a["weight"])
    summary = (
        f"Averaged over the {pred_len}-session forecast, {recent_share * 100:.0f}% of the model's "
        f"attention fell on the most recent {recent_k} trading days in the input window, most heavily "
        f"on {top['date']} ({top['weight'] * 100:.0f}%). "
        f"{historical_reliance * 100:.0f}% of attention referenced actual historical prices; the rest "
        f"referenced the model's own earlier steps in this forecast."
    )

    def to_records(dates, frame):
        out = []
        for date, (_, row) in zip(dates, frame.iterrows()):
            out.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
            })
        return out

    return {
        "symbol": symbol,
        "ticker": ticker,
        "last_close": float(hist["Close"].iloc[-1]),
        "last_date": last_date.strftime("%Y-%m-%d"),
        "historical": to_records(hist["Date"], candles_df),
        "forecast": to_records(future_dates, pred_df.reset_index(drop=True)),
        "explanation": {
            "summary": summary,
            "historical_reliance": historical_reliance,
            "attention": attention,
        },
    }
