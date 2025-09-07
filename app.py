import streamlit as st
import requests
import time
import pandas as pd

st.set_page_config(page_title="💎 Golden Sniper Pro BTC/USDT Predictor", layout="wide")

# Placeholders
price_placeholder = st.empty()
signal_placeholder = st.empty()
confidence_placeholder = st.empty()

# Fetch recent BTC/USDT 1m candles from Binance.US
def fetch_binance_candles():
    url = "https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])
    df["Close"] = df["Close"].astype(float)
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    return df

# Calculate EMA
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# Calculate RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Determine trend & confidence
def get_sniper_signal(df):
    close = df["Close"]
    ema_fast = calculate_ema(close, 9).iloc[-1]
    ema_slow = calculate_ema(close, 21).iloc[-1]
    rsi = calculate_rsi(close).iloc[-1]

    trend = "WAIT"
    confidence = 0.0

    if ema_fast > ema_slow and rsi > 50:
        trend = "UP"
        confidence = min((rsi-50)/50 + 0.5, 1.0)
    elif ema_fast < ema_slow and rsi < 50:
        trend = "DOWN"
        confidence = min((50-rsi)/50 + 0.5, 1.0)

    return trend, confidence, rsi

# Live update loop
while True:
    try:
        df = fetch_binance_candles()
        price = df["Close"].iloc[-1]
        trend, confidence, rsi = get_sniper_signal(df)

        # Update placeholders
        price_placeholder.markdown(f"**Price:** ${price:,.2f}")
        signal_placeholder.markdown(f"**Trend Signal:** {trend} (RSI: {rsi:.1f})")
        confidence_placeholder.markdown(f"**Confidence:** {int(confidence*100)}%")
        
        time.sleep(1)  # Update every second
    except Exception as e:
        st.error(f"Error: {e}")
        time.sleep(5)
