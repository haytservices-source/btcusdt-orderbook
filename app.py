import streamlit as st
import websocket, json, threading
import pandas as pd

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
if "bids" not in st.session_state:
    st.session_state.bids = {}
if "asks" not in st.session_state:
    st.session_state.asks = {}
if "connected" not in st.session_state:
    st.session_state.connected = False

# WebSocket callbacks
def on_message(ws, message):
    data = json.loads(message)
    if "p" in data:  # trade event
        st.session_state.latest_price = float(data["p"])
        st.session_state.connected = True

# Start WebSocket thread
def start_ws():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.us:9443/ws/btcusdt@trade",
        on_message=on_message
    )
    ws.run_forever()

if "ws_thread" not in st.session_state:
    ws_thread = threading.Thread(target=start_ws, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread = ws_thread

# Auto-refresh every 1 second
st_autorefresh = st.experimental_singleton(lambda: None)  # Dummy placeholder
st.experimental_rerun()

# Display
if not st.session_state.connected:
    price_placeholder.text("Connecting to Binance US WebSocket…")
    buy_strength_placeholder.text("Buyers: Loading…")
    sell_strength_placeholder.text("Sellers: Loading…")
    order_book_placeholder.text("Order Book: Loading…")
    volume_placeholder.text("Volume: Loading…")
else:
    price_placeholder.markdown(f"### Price: ${st.session_state.latest_price:,.2f}")
