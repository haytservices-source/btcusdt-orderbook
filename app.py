import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="💎 Golden Sniper Pro BTC/USDT Predictor", layout="wide")
st.title("💎 Golden Sniper Pro BTC/USDT Predictor")

# ---------------------------
# PLACEHOLDERS
# ---------------------------
price_placeholder = st.empty()
trend_placeholder = st.empty()
confidence_placeholder = st.empty()
chart_placeholder = st.empty()

# ---------------------------
# FUNCTIONS
# ---------------------------

BINANCE_URL = "https://api.binance.us/api/v3/klines"

def get_klines(symbol="BTCUSDT", interval="1m", limit=100):
    try:
        url = f"{BINANCE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url).json()
        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume","close_time",
            "quote_asset_volume","number_of_trades","taker_buy_base",
            "taker_buy_quote","ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        print("Error fetching klines:", e)
        return pd.DataFrame()

def calculate_indicators(df):
    # EMA
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # Volume spike
    df["vol_avg"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > 1.5 * df["vol_avg"]
    return df

def analyze_trend(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    trend = "WAIT…"
    confidence = 0.0

    # EMA trend
    if last["EMA20"] > last["EMA50"]:
        trend = "UP"
        confidence += 0.3
    elif last["EMA20"] < last["EMA50"]:
        trend = "DOWN"
        confidence += 0.3

    # RSI filter
    if last["RSI"] < 30:
        trend = "UP (RSI oversold)"
        confidence += 0.2
    elif last["RSI"] > 70:
        trend = "DOWN (RSI overbought)"
        confidence += 0.2

    # Volume spike confirmation
    if last["vol_spike"]:
        confidence += 0.3

    confidence = min(confidence, 1.0)
    return trend, confidence

def get_current_price(symbol="BTCUSDT"):
    try:
        url = f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}"
        data = requests.get(url).json()
        return float(data["price"])
    except:
        return None

# ---------------------------
# LIVE LOOP
# ---------------------------
while True:
    # Fetch data
    df = get_klines()
    df = calculate_indicators(df)
    trend, confidence = analyze_trend(df)
    price = get_current_price()

    # Update dashboard
    price_placeholder.metric("Price", f"${price:.2f}" if price else "Fetching…")
    trend_placeholder.metric("Trend Signal", trend)
    confidence_placeholder.metric("Confidence", f"{confidence*100:.0f}%")
    
    # Update candlestick chart
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])

        # Add signal arrow
        arrow = "➡️"
        if "UP" in trend:
            arrow = "⬆️"
        elif "DOWN" in trend:
            arrow = "⬇️"
        elif "REVERSAL" in trend:
            arrow = "⚡"

        fig.add_annotation(
            x=df["time"].iloc[-1],
            y=df["close"].iloc[-1],
            text=arrow,
            showarrow=False,
            font=dict(size=25)
        )

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=550,
            margin=dict(l=10, r=10, t=10, b=10)
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)

    time.sleep(1)
