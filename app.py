import streamlit as st
import websocket
import json
import threading
import pandas as pd
import time

st.set_page_config(page_title="💎 BTC/USDT Buyer vs Seller Dashboard", layout="wide")

# Placeholders
price_placeholder = st.empty()
buy_strength_placeholder = st.empty()
sell_strength_placeholder = st.empty()
order_book_placeholder = st.empty()
volume_placeholder = st.empty()

# Shared state
if "latest_price" not in st.session_state:
    st.session_state.latest_price = 0
if "bids_total" not in st.session_state:
    st.session_state.bids_total = 0
if "asks_total" not in st.session_state:
    st.session_state.asks_total = 0

# WebSocket callbacks
def on_message(ws, message):
    data = json.loads(message)
    if "p" in data:
        st.session_state.latest_price = float(data["p"])
        st.session_state.bids_total += float(data["q"]) if data["m"] == False else 0
        st.session_state.asks_total += float(data["q"]) if data["m"] == True else 0

def on_error(ws, error):
    print("WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.us:9443/ws/btcusdt@trade",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

# Start WebSocket in background
if "ws_thread" not in st.session_state:
    ws_thread = threading.Thread(target=start_ws, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread = ws_thread

# Live dashboard update
while True:
    price_placeholder.markdown(f"### Price: ${st.session_state.latest_price:,.2f}")
    buy_strength_placeholder.markdown(f"Buyers Strength: {st.session_state.bids_total:.4f} BTC")
    sell_strength_placeholder.markdown(f"Sellers Strength: {st.session_state.asks_total:.4f} BTC")
    volume_placeholder.markdown(f"Volume (last update): {st.session_state.bids_total + st.session_state.asks_total:.4f} BTC")
    
    time.sleep(1)
