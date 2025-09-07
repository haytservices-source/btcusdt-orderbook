import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
import random

st.set_page_config(page_title="BTCUSDT Predictor", layout="wide")
st.title("BTC/USDT Live Predictor")

# --- Fake predictor (replace later with WebSocket logic) ---
def get_prediction(price):
    direction = random.choice(["UP 📈", "DOWN 📉"])
    score = random.random()
    return {"price": price, "prediction": direction, "score": score}

# --- Get live BTCUSDT data from Binance (1m candles) ---
def get_candles():
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 50}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and "code" in data:
            st.error(f"Binance error: {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        return df

    except Exception as e:
        st.error(f"Error fetching candles: {e}")
        return pd.DataFrame()

# --- Live updating UI ---
placeholder = st.empty()

while True:
    df = get_candles()
    if df.empty:
        st.warning("⚠️ Still no data, Binance might be blocking API")
        time.sleep(5)
        continue

    last_price = df["close"].iloc[-1]
    prediction = get_prediction(last_price)

    with placeholder.container():
        # Show metrics
        st.subheader("Prediction")
        col1, col2, col3 = st.columns(3)
        col1.metric("Price", f"{prediction['price']:.2f}")
        col2.metric("Prediction", prediction["prediction"])
        col3.metric("Confidence", f"{prediction['score']:.2f}")

        # Show chart
        st.subheader("Live BTC/USDT Chart (1m)")
        fig = go.Figure(data=[go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    time.sleep(5)  # update every 5 seconds
