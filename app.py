import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from streamlit_autorefresh import st_autorefresh

# --- Page Setup ---
st.set_page_config(page_title="Golden Sniper Pro BTC/USDT Predictor", layout="wide")
st.title("💎 Golden Sniper Pro BTC/USDT Predictor (Binance.US)")

# --- Auto-refresh every 1 second ---
st_autorefresh(interval=1000, key="refresh")

# --- Fetch Candles from Binance.US ---
@st.cache_data(ttl=2)
def get_candles(interval="1m", limit=150):
    try:
        r = requests.get("https://api.binance.us/api/v3/klines",
                         params={"symbol":"BTCUSDT","interval":interval,"limit":limit}, timeout=5)
        data = r.json()
        if not data or len(data) < 20:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        return df
    except:
        return pd.DataFrame()

# --- Smart Money Concepts + Trend + Volume ---
def analyze_smc(df):
    if df.empty or len(df)<20:
        return "WAIT…", 0.0

    # EMA Trend
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    ema_trend = "UP" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "DOWN"

    # RSI Momentum
    delta = df["close"].diff()
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    rsi_trend = "UP" if df["RSI"].iloc[-1] > 50 else "DOWN"

    # Volume Spike
    vol_mean = df["volume"].rolling(20).mean()
    vol_spike = df["volume"].iloc[-1] > 2 * vol_mean.iloc[-1]

    # BOS / CHOCH Detection (simplified)
    last_high = df["high"].iloc[-2]
    last_low = df["low"].iloc[-2]
    curr_high = df["high"].iloc[-1]
    curr_low = df["low"].iloc[-1]

    bos_up = curr_high > last_high
    bos_down = curr_low < last_low
    choch = (bos_up and ema_trend=="DOWN") or (bos_down and ema_trend=="UP")

    # Combine Signals
    last_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    confidence = 0.5
    signal = "SIDEWAYS ➡️"

    if ema_trend=="UP" and rsi_trend=="UP" and vol_spike and bos_up:
        signal = "STRONG UP 📈"
        confidence = 0.95
    elif ema_trend=="DOWN" and rsi_trend=="DOWN" and vol_spike and bos_down:
        signal = "STRONG DOWN 📉"
        confidence = 0.95
    elif bos_up:
        signal = "UP 📈"
        confidence = 0.7
    elif bos_down:
        signal = "DOWN 📉"
        confidence = 0.7
    elif choch:
        signal = "REVERSAL ⚡"
        confidence = 0.85

    return signal, confidence

# --- Multi-Timeframe Filtering ---
def get_multi_tf_signal():
    df1 = get_candles("1m")
    df5 = get_candles("5m")
    df15 = get_candles("15m")

    signals = []
    confidences = []
    df_use = df1  # fallback to 1m

    for df_tf in [df1, df5, df15]:
        if not df_tf.empty:
            s, c = analyze_smc(df_tf)
            signals.append(s)
            confidences.append(c)
            df_use = df_tf  # last valid df

    if not signals:
        return "WAIT…", 0.0, df1

    # Most frequent signal
    trend = pd.Series(signals).value_counts().idxmax()
    confidence = np.mean(confidences)
    return trend, confidence, df_use

# --- Live Price ---
@st.cache_data(ttl=1)
def get_current_price():
    try:
        r = requests.get("https://api.binance.us/api/v3/ticker/price", params={"symbol":"BTCUSDT"}, timeout=5)
        return float(r.json()["price"])
    except:
        return None

# --- Main ---
trend, confidence, df = get_multi_tf_signal()
current_price = get_current_price()

# --- Display Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Price", f"{current_price:.2f}" if current_price else "Fetching…")
col2.metric("Trend Signal", trend)
col3.metric("Confidence", f"{confidence:.2f}")

# --- Candlestick Chart ---
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

    # Prediction Arrow
    arrow_text = "➡️"
    if "UP" in trend:
        arrow_text = "⬆️"
    elif "DOWN" in trend:
        arrow_text = "⬇️"
    elif "REVERSAL" in trend:
        arrow_text = "⚡"

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
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.caption("💡 Multi-Timeframe + SMC + EMA + RSI + Volume Spike Analysis | High-Probability Sniper Signals | Data: Binance.US API")
