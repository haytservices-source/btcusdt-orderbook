import streamlit as st
import requests
import websocket
import threading
import time
import json

st.set_page_config(page_title="💎 BTC/USDT Buyer vs Seller Dashboard", layout="wide")

# Placeholders
price_placeholder = st.empty()
buyers_placeholder = st.empty()
sellers_placeholder = st.empty()
volume_placeholder = st.empty()
order_book_placeholder = st.empty()

# Shared state
if "price" not in st.session_state:
    st.session_state.price = 0
if "buy_volume" not in st.session_state:
    st.session_state.buy_volume = 0
if "sell_volume" not in st.session_state:
    st.session_state.sell_volume = 0
if "order_book" not in st.session_state:
    st.session_state.order_book = []

# Function to fetch initial order book snapshot
def fetch_order_book():
    url = "https://api.binance.us/api/v3/depth?symbol=BTCUSDT&limit=5"
    data = requests.get(url).json()
    st.session_state.order_book = data
    st.session_state.buy_volume = sum(float(bid[1]) for bid in data['bids'])
    st.session_state.sell_volume = sum(float(ask[1]) for ask in data['asks'])

# WebSocket callbacks
def on_message(ws, message):
    data = json.loads(message)
    st.session_state.price = float(data['p'])
    if data['m']:  # sell
        st.session_state.sell_volume += float(data['q'])
    else:          # buy
        st.session_state.buy_volume += float(data['q'])

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

# Fetch initial order book
fetch_order_book()

# Start WebSocket in background
if "ws_thread" not in st.session_state:
    ws_thread = threading.Thread(target=start_ws, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread = ws_thread

# Live dashboard
while True:
    price_placeholder.markdown(f"### Price: ${st.session_state.price:,.2f}")
    buyers_placeholder.markdown(f"**Buyers Strength:** {st.session_state.buy_volume:.4f} BTC")
    sellers_placeholder.markdown(f"**Sellers Strength:** {st.session_state.sell_volume:.4f} BTC")
    volume_placeholder.markdown(f"**Total Volume:** {(st.session_state.buy_volume + st.session_state.sell_volume):.4f} BTC")
    
    order_book_md = "### Top 5 Order Book Levels\n\n| Bid Price | Bid Qty | Ask Price | Ask Qty |\n|---|---|---|---|\n"
    for i in range(5):
        bid = st.session_state.order_book['bids'][i]
        ask = st.session_state.order_book['asks'][i]
        order_book_md += f"| {bid[0]} | {bid[1]} | {ask[0]} | {ask[1]} |\n"
    order_book_placeholder.markdown(order_book_md)
    
    time.sleep(1)
