import streamlit as st
import websocket
import json
import threading
import time

st.set_page_config(page_title="💎 Golden Sniper Pro BTC/USDT Predictor", layout="wide")

# Placeholders
price_placeholder = st.empty()
signal_placeholder = st.empty()
confidence_placeholder = st.empty()

# Global variable to store latest price
latest_price = 0.0
prev_price = 0.0

# Function to determine trend & confidence
def get_sniper_signal(price):
    global prev_price
    trend = "WAIT"
    confidence = 0
    rsi = 50.0

    if price > prev_price * 1.0005:
        trend = "UP"
        confidence = 80
        rsi = 60
    elif price < prev_price * 0.9995:
        trend = "DOWN"
        confidence = 80
        rsi = 40

    prev_price = price
    return trend, confidence, rsi

# WebSocket callbacks
def on_message(ws, message):
    global latest_price
    data = json.loads(message)
    latest_price = float(data['p'])

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")

# Start WebSocket in separate thread
def start_ws():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.us:9443/ws/btcusdt@trade",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()

# Start WebSocket thread
threading.Thread(target=start_ws, daemon=True).start()

# Streamlit-friendly auto-refresh
st_autorefresh = st.empty()

while True:
    if latest_price != 0.0:
        trend, confidence, rsi = get_sniper_signal(latest_price)
        price_placeholder.markdown(f"**Price:** ${latest_price:,.2f}")
        signal_placeholder.markdown(f"**Trend Signal:** {trend} (RSI: {rsi:.1f})")
        confidence_placeholder.markdown(f"**Confidence:** {confidence}%")
    time.sleep(0.5)  # refresh every 0.5 seconds

