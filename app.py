import streamlit as st
import websocket, json
import threading

st.set_page_config(page_title="💎 Golden Sniper Pro BTC/USDT Predictor", layout="wide")
price_placeholder = st.empty()
trend_placeholder = st.empty()
conf_placeholder = st.empty()

# Shared variable
st.session_state.latest_price = 0

def on_message(ws, message):
    data = json.loads(message)
    price = float(data['p'])
    st.session_state.latest_price = price

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.us:9443/ws/btcusdt@trade",
        on_message=on_message
    )
    ws.run_forever()

# Start WebSocket in background thread
if "ws_thread" not in st.session_state:
    ws_thread = threading.Thread(target=start_ws, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread = ws_thread

# Display loop
import time
while True:
    price = st.session_state.latest_price
    if price == 0:
        price_placeholder.text("Price: Fetching…")
        trend_placeholder.text("Trend Signal: WAIT…")
        conf_placeholder.text("Confidence: 0%")
    else:
        # Simple trend logic
        if "prev_price" not in st.session_state:
            st.session_state.prev_price = price
        trend = "WAIT"
        confidence = 50
        if price > st.session_state.prev_price * 1.0005:
            trend = "UP"
            confidence = 80
        elif price < st.session_state.prev_price * 0.9995:
            trend = "DOWN"
            confidence = 80
        st.session_state.prev_price = price

        price_placeholder.text(f"Price: ${price:,.2f}")
        trend_placeholder.text(f"Trend Signal: {trend}")
        conf_placeholder.text(f"Confidence: {confidence}%")
    time.sleep(0.1)
