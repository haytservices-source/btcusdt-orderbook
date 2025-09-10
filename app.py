import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every 2 seconds ---
st_autorefresh(interval=2000, key="refresh")

st.set_page_config(page_title="BTC/USDT Smart Multi-Timeframe Order Flow", layout="wide")
st.title("📊 BTC/USDT Advanced Smart Signals (Automatic Multi-Timeframe)")

exchange = "binance.us"
base_url = "https://api.binance.us"

# --- Fetch Candles ---
def get_candles(interval="1m", limit=50):
    url = f"{base_url}/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
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
        st.error(f"Candles fetch error: {e}")
        return None

# --- Fetch Order Book (1m only) ---
def get_orderbook(limit=200):
    url = f"{base_url}/api/v3/depth"
    params = {"symbol": "BTCUSDT", "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        data = res.json()
        bids = pd.DataFrame(data["bids"], columns=["price","qty"], dtype=float)
        asks = pd.DataFrame(data["asks"], columns=["price","qty"], dtype=float)
        return bids, asks
    except Exception as e:
        st.error(f"Order book fetch error: {e}")
        return None, None

# --- Candle Trend ---
def candle_trend(df, lookback=5):
    if len(df) < lookback:
        return 0
    recent_close = df['close'].iloc[-1]
    past_close = df['close'].iloc[-lookback]
    return 1 if recent_close > past_close else -1 if recent_close < past_close else 0

# --- Volume Spike ---
def volume_spike(df, multiplier=1.5):
    avg_vol = df['volume'].iloc[-10:-1].mean()
    recent_vol = df['volume'].iloc[-1]
    return recent_vol > avg_vol * multiplier

# --- Order Book Pressure ---
def calculate_pressure(bids, asks):
    buy_pressure = (bids["price"] * bids["qty"]).sum()
    sell_pressure = (asks["price"] * asks["qty"]).sum()
    total = buy_pressure + sell_pressure
    confidence = abs(buy_pressure - sell_pressure) / total if total != 0 else 0
    return buy_pressure, sell_pressure, confidence

# --- Generate Smart Signal ---
def generate_signal(trends, buy_pressure, sell_pressure, vol_spike):
    # Count bullish/bearish trends
    bullish = trends.count(1)
    bearish = trends.count(-1)

    if bullish > bearish and buy_pressure > sell_pressure and vol_spike:
        return "BUY ⬆️"
    elif bearish > bullish and sell_pressure > buy_pressure and vol_spike:
        return "SELL ⬇️"
    else:
        return "SIDEWAYS ➡️"

# --- Main Execution ---
# Multi-timeframes: 1m, 5m, 15m
timeframes = ["1m", "5m", "15m"]
trends = []
vol_spikes = []

for tf in timeframes:
    df = get_candles(interval=tf)
    if df is not None:
        trends.append(candle_trend(df))
        vol_spikes.append(volume_spike(df))
    else:
        trends.append(0)
        vol_spikes.append(False)

# Order book (1m)
bids, asks = get_orderbook()
if bids is not None and asks is not None:
    buy_pressure, sell_pressure, confidence = calculate_pressure(bids, asks)
else:
    buy_pressure = sell_pressure = confidence = 0

# Combine volume spike from all timeframes
vol_spike_combined = any(vol_spikes)

# Generate final smart signal
signal = generate_signal(trends, buy_pressure, sell_pressure, vol_spike_combined)

# --- Display ---
st.subheader("📈 Smart Multi-Timeframe Signal")
st.markdown(f"**Signal:** {signal}")
st.markdown(f"**Confidence:** {confidence:.2f}")
trend_text = ["Bullish" if t==1 else "Bearish" if t==-1 else "Neutral" for t in trends]
st.markdown(f"**Trends (1m / 5m / 15m):** {trend_text[0]} | {trend_text[1]} | {trend_text[2]}")
st.markdown(f"**Volume Spike Detected:** {'Yes' if vol_spike_combined else 'No'}")

# --- Plot 1m Candlestick Chart ---
candles_1m = get_candles("1m")
if candles_1m is not None:
    fig = go.Figure(data=[go.Candlestick(
        x=candles_1m['time'],
        open=candles_1m['open'],
        high=candles_1m['high'],
        low=candles_1m['low'],
        close=candles_1m['close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig.update_layout(title="BTC/USDT 1m Candles", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- Plot Order Book ---
if bids is not None and asks is not None:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=bids['qty'], x=bids['price'], name='Bids', marker_color='green'))
    fig2.add_trace(go.Bar(y=asks['qty'], x=asks['price'], name='Asks', marker_color='red'))
    fig2.update_layout(title="Order Book (Top 200 Levels)", barmode='overlay', xaxis_title="Price", yaxis_title="Quantity")
    st.plotly_chart(fig2, use_container_width=True)
