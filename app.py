import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- Page Config ---
st.set_page_config(page_title="BTCUSDT Live Predictor", layout="wide")
st.title("BTC/USDT Live Predictor")

# --- Auto-refresh every 1 second ---
st_autorefresh(interval=1000, key="refresh")

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

# --- Fetch 1-minute candles ---
@st.cache_data(ttl=2)
def get_candles():
    urls = [
        "https://api.binance.com/api/v3/klines",
        "https://fapi.binance.com/fapi/v1/klines",
        "https://api.binance.us/api/v3/klines"
    ]
    for url in urls:
        try:
            params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 50}
            r = requests.get(url, params=params, timeout=5)
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                continue
            df = pd.DataFrame(data, columns=[
                "time","open","high","low","close","volume",
                "close_time","qav","num_trades","taker_base","taker_quote","ignore"
            ])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
            df.attrs["source"] = url
            return df
        except:
            continue
    return pd.DataFrame()

# --- Fetch live current price ---
@st.cache_data(ttl=1)
def get_current_price():
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol":"BTCUSDT"}, timeout=5)
        return float(r.json()["price"])
    except:
        return None

# --- Main UI ---
df = get_candles()
current_price = get_current_price()

if df.empty:
    st.warning("⚠️ No candle data received from Binance. Retrying…")
else:
    last_candle_close = df["close"].iloc[-1]
    prediction, confidence = get_prediction(df["close"].tolist())
    source = df.attrs.get("source", "Unknown")

    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"{current_price:.2f}" if current_price else f"{last_candle_close:.2f}")
    col2.metric("Prediction", prediction)
    col3.metric("Confidence", f"{confidence:.2f}")
    st.caption(f"✅ Candle Data Source: {source} | Live Price Source: Binance API")

    # --- Candlestick chart ---
    st.subheader("Live BTC/USDT 1m Candles")
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])

    # Add last candle prediction arrow
    if prediction.startswith("UP"):
        fig.add_annotation(x=df["time"].iloc[-1], y=df["close"].iloc[-1],
                           text="⬆️", showarrow=False, font=dict(size=20))
    elif prediction.startswith("DOWN"):
        fig.add_annotation(x=df["time"].iloc[-1], y=df["close"].iloc[-1],
                           text="⬇️", showarrow=False, font=dict(size=20))
    else:
        fig.add_annotation(x=df["time"].iloc[-1], y=df["close"].iloc[-1],
                           text="➡️", showarrow=False, font=dict(size=20))

    # Optional: Add tiny arrows on last few candles for quick trend
    for i in range(-5, 0):
        if df["close"].iloc[i] > df["open"].iloc[i]:
            fig.add_annotation(x=df["time"].iloc[i], y=df["high"].iloc[i],
                               text="⬆️", showarrow=False, font=dict(size=12))
        elif df["close"].iloc[i] < df["open"].iloc[i]:
            fig.add_annotation(x=df["time"].iloc[i], y=df["low"].iloc[i],
                               text="⬇️", showarrow=False, font=dict(size=12))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
