import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every 2 seconds ---
st_autorefresh(interval=2000, key="refresh")

st.set_page_config(page_title="BTC/USDT Advanced Order Flow", layout="wide")
st.title("📊 BTC/USDT Advanced Order Flow Dashboard")

# --- Exchange ---
base_url = "https://api.binance.us"

# --- Fetch Order Book ---
def get_orderbook(symbol="BTCUSDT", limit=200):
    url = f"{base_url}/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        data = res.json()
        bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)
        return bids, asks
    except Exception as e:
        st.error(f"❌ Order book fetch error: {e}")
        return None, None

# --- Fetch Candlestick Data ---
def get_candles(symbol="BTCUSDT", interval="1m", limit=50):
    url = f"{base_url}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        data = res.json()
        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume",
            "c","q","n","taker_base","taker_quote","ignore"
        ], dtype=float)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df[["time","open","high","low","close","volume"]]
    except Exception as e:
        st.error(f"❌ Candles fetch error: {e}")
        return None

# --- Buy/Sell Pressure Calculation ---
def calculate_pressure(bids, asks):
    if bids is None or asks is None:
        return None, None
    buy_pressure = (bids['qty'] * bids['price']).sum()
    sell_pressure = (asks['qty'] * asks['price']).sum()
    return buy_pressure, sell_pressure

# --- Secret Strategy Prediction ---
def secret_strategy(buy_pressure, sell_pressure):
    if buy_pressure is None or sell_pressure is None:
        return "NO DATA", 0
    total = buy_pressure + sell_pressure
    confidence = abs(buy_pressure - sell_pressure) / total
    if buy_pressure > sell_pressure * 1.05:
        return "BUY ⬆️", confidence
    elif sell_pressure > buy_pressure * 1.05:
        return "SELL ⬇️", confidence
    else:
        return "SIDEWAYS ➡️", confidence

# --- Fetch Data ---
bids, asks = get_orderbook()
candles = get_candles()

if bids is not None and asks is not None:
    buy_pressure, sell_pressure = calculate_pressure(bids, asks)
    signal, confidence = secret_strategy(buy_pressure, sell_pressure)
else:
    buy_pressure = sell_pressure = signal = confidence = None

# --- Display Dashboard ---
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("🔹 Market Pressure")
    if buy_pressure is not None:
        st.metric("Buy Pressure", f"{buy_pressure:,.0f}")
        st.metric("Sell Pressure", f"{sell_pressure:,.0f}")
        st.metric("Signal", f"{signal} ({confidence:.2f})")
    else:
        st.info("Waiting for data...")

with col2:
    st.subheader("📈 Live Candlestick Chart")
    if candles is not None and not candles.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=candles['time'],
            open=candles['open'],
            high=candles['high'],
            low=candles['low'],
            close=candles['close'],
            name="Candles"
        )])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            margin=dict(l=10,r=10,t=30,b=10),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for candle data...")

st.markdown("---")
st.markdown("⚡ Dashboard updates every 2 seconds. Data source: [Binance.US](https://www.binance.us)")
