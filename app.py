import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 2 seconds
st_autorefresh(interval=2000, key="refresh")

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

# --- Get candles ---
def get_candles():
    urls = [
        "https://api.binance.com/api/v3/klines",     
        "https://fapi.binance.com/fapi/v1/klines",   
        "https://api.binance.us/api/v3/klines"       
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
            df.attrs["source"] = url
            return df
        except Exception:
            continue
    return pd.DataFrame()

# --- Main UI ---
df = get_candles()

if df.empty:
    st.warning("⚠️ No data received from Binance (Spot, Futures, US). Retrying…")
else:
    last_price = df["close"].iloc[-1]
    prediction, confidence = get_prediction(df["close"].tolist())
    source = df.attrs.get("source", "Unknown")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"{last_price:.2f}")
    col2.metric("Prediction", prediction)
    col3.metric("Confidence", f"{confidence:.2f}")
    st.caption(f"✅ Data source: {source}")

    # Candlestick chart
    st.subheader("Live BTC/USDT 1m Candles")
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
`
