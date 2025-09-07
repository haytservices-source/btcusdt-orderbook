import streamlit as st
import websocket, json, threading
import pandas as pd
from collections import deque
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
if "bids" not in st.session_state:
    st.session_state.bids = {}
if "asks" not in st.session_state:
    st.session_state.asks = {}

# WebSocket callbacks
def on_message(ws, message):
    data = json.loads(message)
    event_type = data.get("e")

    if event_type == "trade":
        st.session_state.latest_price = float(data["p"])
    elif event_type == "depthUpdate":
        # Update bids and asks
        for bid in data["b"]:
            price, qty = float(bid[0]), float(bid[1])
            if qty == 0 and price in st.session_state.bids:
                del st.session_state.bids[price]
            else:
                st.session_state.bids[price] = qty

        for ask in data["a"]:
            price, qty = float(ask[0]), float(ask[1])
            if qty == 0 and price in st.session_state.asks:
                del st.session_state.asks[price]
            else:
                st.session_state.asks[price] = qty

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.us:9443/ws/btcusdt@depth",
        on_message=on_message
    )
    ws.run_forever()

# Start WebSocket thread
if "ws_thread" not in st.session_state:
    ws_thread = threading.Thread(target=start_ws, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread = ws_thread

# Utility function to calculate strength
def calculate_strength(book):
    total_qty = sum(book.values())
    top_qty = sum(list(book.values())[:5])
    return top_qty, total_qty

# Display loop
while True:
    price = st.session_state.latest_price
    bids = st.session_state.bids
    asks = st.session_state.asks

    if price == 0 or len(bids) == 0 or len(asks) == 0:
        price_placeholder.text("Price: Fetching…")
        buy_strength_placeholder.text("Buyers: Loading…")
        sell_strength_placeholder.text("Sellers: Loading…")
        order_book_placeholder.text("Order Book: Loading…")
        volume_placeholder.text("Volume: Loading…")
    else:
        # Price
        price_placeholder.markdown(f"### Price: ${price:,.2f}")

        # Strength
        top_buy, total_buy = calculate_strength(dict(sorted(bids.items(), reverse=True)))
        top_sell, total_sell = calculate_strength(dict(sorted(asks.items())))
        buy_strength_placeholder.text(f"Buy Strength → Top5: {top_buy:.2f} | Total: {total_buy:.2f}")
        sell_strength_placeholder.text(f"Sell Strength → Top5: {top_sell:.2f} | Total: {total_sell:.2f}")

        # Order Book scale
        order_book_placeholder.text(
            f"Top 5 Bids:\n{pd.DataFrame(sorted(bids.items(), reverse=True)[:5], columns=['Price','Qty'])}\n\n"
            f"Top 5 Asks:\n{pd.DataFrame(sorted(asks.items())[:5], columns=['Price','Qty'])}"
        )

        # Volume pushing
        buy_push = top_buy / (top_buy + top_sell) * 100
        sell_push = top_sell / (top_buy + top_sell) * 100
        volume_placeholder.text(f"Buy Pressure: {buy_push:.1f}% | Sell Pressure: {sell_push:.1f}%")

    time.sleep(0.5)
