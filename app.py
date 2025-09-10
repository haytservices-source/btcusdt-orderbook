import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ta
import plotly.graph_objects as go

# -------------------------------
# Generate synthetic BTC/USDT data
# -------------------------------
np.random.seed(42)
n = 500
prices = np.cumsum(np.random.randn(n)) + 27000
volume = np.random.randint(100, 1000, n)

df = pd.DataFrame({
    "close": prices,
    "open": prices + np.random.randn(n),
    "high": prices + np.random.rand(n) * 10,
    "low": prices - np.random.rand(n) * 10,
    "volume": volume
})

# -------------------------------
# Add technical indicators
# -------------------------------
df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()
df["macd"] = ta.trend.MACD(df["close"]).macd()
df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()

# -------------------------------
# Target variable (next move)
# -------------------------------
df["future"] = df["close"].shift(-1)
df["direction"] = np.where(df["future"] > df["close"], 1,
                  np.where(df["future"] < df["close"], -1, 0))
df = df.dropna()

features = ["rsi", "ema_fast", "ema_slow", "macd", "macd_signal", "volume"]
X = df[features]
y = df["direction"]

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -------------------------------
# Streamlit Dashboard
# -------------------------------
st.set_page_config(page_title="BTC/USDT ML Predictor", layout="wide")
st.title("📈 BTC/USDT ML Predictor")

# Show accuracy
st.metric("Model Accuracy (test data)", f"{acc*100:.2f}%")

# Last prediction
latest_features = X.iloc[-1:].values
latest_pred = model.predict(latest_features)[0]
latest_prob = model.predict_proba(latest_features)[0]

direction_map = {1: "📈 BUY", -1: "📉 SELL", 0: "➡️ SIDEWAYS"}
st.subheader("Latest Market Prediction")
st.write(f"**Signal:** {direction_map[latest_pred]}")
st.write(f"**Confidence:** {np.max(latest_prob)*100:.2f}%")

# Candle chart
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
)])
fig.update_layout(title="BTC/USDT Candlestick (Synthetic Data)", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
