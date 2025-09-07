import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time

# ------------------------------
# Helper Functions
# ------------------------------
def get_current_price():
    url = "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT"
    try:
        data = requests.get(url, timeout=5).json()
        return float(data['price'])
    except:
        return None

def get_candles(interval="1m", limit=50):
    url = f"https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
    try:
        data = requests.get(url, timeout=5).json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
        ])
        df['time'] = pd.to_datetime(df['open_time'], unit='ms')
        df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
        return df
    except:
        return pd.DataFrame()

def analyze_trend(df):
    if df.empty:
        return "WAIT…", 0.0
    # Example logic: RSI + EMA + Volume spike
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean() / df['close'].pct_change().rolling(14).std()))
    last_close = df['close'].iloc[-1]
    trend = "WAIT…"
    confidence = 0.0

    if last_close > df['ema20'].iloc[-1] and df['ema20'].iloc[-1] > df['ema50'].iloc[-1] and df['rsi'].iloc[-1] < 70:
        trend = "UP"
        confidence = 0.8
    elif last_close < df['ema20'].iloc[-1] and df['ema20'].iloc[-1] < df['ema50'].iloc[-1] and df['rsi'].iloc[-1] > 30:
        trend = "DOWN"
        confidence = 0.7
    else:
        trend = "SIDEWAYS"
        confidence = 0.5
    return trend, confidence

# ------------------------------
# Streamlit Layout
# ------------------------------
st.set_page_config(page_title="💎 Golden Sniper Pro BTC/USDT Predictor", layout="wide")
st.title("💎 Golden Sniper Pro BTC/USDT Predictor")

# Placeholders
price_placeholder = st.empty()
trend_placeholder = st.empty()
confidence_placeholder = st.empty()
chart_placeholder = st.empty()

# ------------------------------
# Live Update Loop
# ------------------------------
while True:
    # Fetch Data
    price = get_current_price()
    df = get_candles()
    trend, confidence = analyze_trend(df)

    # Update Metrics
    price_placeholder.metric("Price", f"${price:.2f}" if price else "Fetching…")
    trend_placeholder.metric("Trend Signal", trend)
    confidence_placeholder.metric("Confidence", f"{int(confidence*100)}%")

    # Update Chart
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

        # Arrow annotation for trend
        arrow_text = "➡️"
        if trend == "UP":
            arrow_text = "⬆️"
        elif trend == "DOWN":
            arrow_text = "⬇️"
        elif trend == "SIDEWAYS":
            arrow_text = "➡️"

        fig.add_annotation(
            x=df["time"].iloc[-1],
            y=df["close"].iloc[-1],
            text=arrow_text,
            showarrow=False,
            font=dict(size=25)
        )

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=550,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    time.sleep(1)  # Update every second
