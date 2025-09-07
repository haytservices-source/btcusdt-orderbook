import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import random

st.set_page_config(page_title="BTCUSDT Predictor", layout="wide")
st.title("BTC/USDT Live Predictor")

# --- Auto-refresh every 5 seconds ---
st_autorefresh(interval=5000, key="btc_refresh")

# --- Fake predictor (replace later with your logic) ---
def get_prediction():
    price = random.uniform(60000, 65000)
    direction = random.choice(["UP 📈", "DOWN 📉"])
    score = random.random()
    return {"price": round(price, 2), "prediction": direction, "score": score}

# --- Get live BTCUSDT data from Binance ---
def get_candles():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 100}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
    return df

# --- Show prediction ---
data = get_prediction()
st.subheader("Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Price", data["price"])
col2.metric("Prediction", data["prediction"])
col3.metric("Confidence", f"{data['score']:.2f}")

# --- Show live candlestick chart ---
st.subheader("Live BTC/USDT Chart (1m candles)")
df = get_candles()

fig = go.Figure(data=[go.Candlestick(
    x=df["time"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
)])
fig.update_layout(xaxis_rangeslider_visible=False, height=500)

st.plotly_chart(fig, use_container_width=True)
