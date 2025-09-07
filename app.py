import streamlit as st
import websocket
import json
import threading
import time
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="BTCUSDT Order Flow Predictor", layout="wide")

SYMBOL = "BTCUSDT"
latest_data = {"bids": None, "asks": None}

# Function to handle incoming WebSocket messages
def on_message(ws, message):
    data = json.loads(message)
    latest_data["bids"] = data.get("bids")
    latest_data["asks"] = data.get("asks")

# Error handling
def on_error(ws, error):
    st.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    st.warning("WebSocket closed")

# Function to run WebSocket
def run_ws():
    url = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth20@100ms"
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()

# Start WebSocket in a separate thread
threading.Thread(target=run_ws, daemon=True).start()

# Auto-refresh every second
st_autorefresh = st.empty()

while True:
    time.sleep(1)
    if latest_data["bids"] is None or latest_data["asks"] is None:
        st.warning("Waiting for live order book data...")
        continue

    bids_df = pd.DataFrame(latest_data["bids"], columns=["Price", "Quantity"]).astype(float)
    asks_df = pd.DataFrame(latest_data["asks"], columns=["Price", "Quantity"]).astype(float)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=bids_df["Price"], y=bids_df["Quantity"], name="Bids", marker_color='green'))
    fig.add_trace(go.Bar(x=asks_df["Price"], y=asks_df["Quantity"], name="Asks", marker_color='red'))

    fig.update_layout(title="BTCUSDT Order Book (Top 20 Levels)", xaxis_title="Price", yaxis_title="Quantity", barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)
    st_autorefresh.empty()
