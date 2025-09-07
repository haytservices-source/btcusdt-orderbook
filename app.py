import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import websocket
import json
import threading
import time

st.set_page_config(page_title="BTCUSDT Live Predictor", layout="wide")
st.title("BTC/USDT Live Predictor")

# Shared data store
price_data = {"price": None, "prices": []}

# --- WebSocket worker ---
def on_message(ws, message):
    global price_data
    msg = json.loads(message)
    price = float(msg["p"])
    price_data["price"] = price
    price_data["prices"].append(price)
    if len(price_data["prices"]) > 200:  # keep last 200 ticks
        price_data["prices"].pop(0)

def on_error(ws, error):
    st.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    st.warning("WebSocket closed")

def run_ws():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws/btcusdt@trade",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

# Start WebSocket thread
threading.Thread(target=run_ws, daemon=True).start()

# --- Simple prediction logic ---
def get_prediction(prices):
    if len(prices) < 2:
        return "Waiting…", 0.0
    if prices[-1] > prices[-2]:
        return "UP 📈", 0.9
    elif prices[-1] < prices[-2]:
        return "DOWN 📉", 0.9
    else:
        return "SIDEWAYS ➡️", 0.5

# --- UI loop ---
placeholder = st.empty()

while True:
    if price_data["price"] is None:
        st.info("Connecting to Binance WebSocket…")
        time.sleep(1)
        continue

    prediction, confidence = get_prediction(price_data["prices"])

    with placeholder.container():
        st.subheader("Prediction")
        col1, col2, col3 = st.columns(3)
        col1.metric("Price", f"{price_data['price']:.2f}")
        col2.metric("Prediction", prediction)
        col3.metric("Confidence", f"{confidence:.2f}")

        # Line chart of recent prices
        st.subheader("Live BTC/USDT Price")
        df = pd.DataFrame(price_data["prices"], columns=["Price"])
        df["Time"] = range(len(df))
        fig = go.Figure(data=go.Scatter(x=df["Time"], y=df["Price"], mode="lines"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    time.sleep(1)  # refresh every 1 second
