import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="BTCUSDT Live Predictor", layout="wide")
st.title("BTC/USDT Live Predictor")

# --- Prediction logic ---
def get_prediction(prices):
    if len(prices) < 2:
        return "Waiting…", 0.0
    if prices[-1] > prices[-2]:
        return "UP 📈", 0.9
    elif prices[-1] < prices[-2]:
        return "DOWN 📉", 0.9
    else:
        return "SIDEWAYS ➡️", 0.5

# --- Get candles from Spot, Futures, or Binance.US ---
def get_candles():
    urls = [
        "https://api.binance.com/api/v3/klines",     # Binance Spot
        "https://fapi.binance.com/fapi/v1/klines",   # Binance Futures
        "https://api.binance.us/api/v3/klines"       # Binance US
    ]
    for url in urls:
        try:
            params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 50}
            r = requests.get(url, params=params, timeout=10)
            data = r.json()

            if not isinstance(data, list) or len(data) == 0:
                continue

            df = pd.DataFrame(data, columns=[
                "time","open","high","low","close","volume",
                "close_time","qav","num_trades","taker_base","taker_quote","ignore"
            ])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)

            # Add source info
            df.attrs["source"] = url
            return df
        except Exception:
            continue
    return pd.DataFrame()

# --- UI loop ---
placeholder = st.empty()
counter = 0  # for unique keys

while True:
    df = get_candles()

    if df.empty:
        with placeholder.container():
            st.warning("⚠️ No data received from Binance (Spot, Futures, US). Retrying…", key=f"warn-{counter}")
        time.sleep(5)
        counter += 1
        continue

    last_price = df["close"].iloc[-1]
    prediction, confidence = get_prediction(df["close"].tolist())
    source = df.attrs.get("source", "Unknown")

    with placeholder.container():
        # Metrics with unique keys
        st.subheader("Prediction")
        col1, col2, col3 = st.columns(3)
        col1.metric("Price", f"{last_price:.2f}", key=f"price-{counter}")
        col2.metric("Prediction", prediction, key=f"pred-{counter}")
        col3.metric("Confidence", f"{confidence:.2f}", key=f"conf-{counter}")

        st.caption(f"✅ Data source: {source}")

        # Candlestick chart with unique key
        st.subheader("Live BTC/USDT 1m Candles")
        fig = go.Figure(data=[go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True, key=f"chart-{counter}")

    time.sleep(2)
    counter += 1
