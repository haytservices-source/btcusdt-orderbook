import streamlit as st
import websocket
import json
import threading

st.title("BTC/USDT (USDⓈ-M Futures) Live Price")

# Placeholder for live price
price_placeholder = st.empty()

# WebSocket message handler
def on_message(ws, message):
    data = json.loads(message)
    price = float(data['p'])  # 'p' is the trade price
    price_placeholder.metric(label="BTC/USDT Price", value=f"${price:,.2f}")

# Run WebSocket in a separate thread
def run_ws():
    ws = websocket.WebSocketApp(
        "wss://fstream.binance.com/ws/btcusdt@trade",
        on_message=on_message
    )
    ws.run_forever()

thread = threading.Thread(target=run_ws)
thread.start()
