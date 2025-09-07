import streamlit as st
import requests
import websocket
import threading
import json

st.set_page_config(page_title="BTC/USDT Buyer vs Seller Dashboard", layout="wide")

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

# Fetch initial order book
def fetch_order_book():
    url = "https://api.binance.us/api/v3/depth?symbol=BTCUSDT&limit=5"
    data = requests.get(url).json()
    st.session_state.order_book = data
    st.session_state.buy_volume = sum(float(bid[1]) for bid in data['bids'])
    st.session_state.sell_volume = sum(float(ask[1]) for ask in data['asks'])

fetch_order_book()

# WebSocket callbacks
def on_message(ws, message):
    data = json.loads(message)
    st.session_state.price = float(data['p'])
    # Update volume for buyers/sellers
    if data['m']:  # trade is a sell
        st.session_state.sell_volume += float(data['q'])
    else:          # trade is a buy
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

# Start WebSocket in background
if "ws_thread" not in st.session_state:
    ws_thread = threading.Thread(target=start_ws, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread = ws_thread

# Dashboard updating in Streamlit main loop
st_autorefresh = st.experimental_rerun

# Use this simple updater instead of while True
def update_dashboard():
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

# Auto refresh every second
st_autorefresh(interval=1000, key="dashboard_refresh")
update_dashboard()
