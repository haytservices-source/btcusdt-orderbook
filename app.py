import streamlit as st
import websocket, json, threading
import pandas as pd
from collections import deque

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
    if data.get("e") == "depthUpdate":
        # Update bids
        for bid in data["b"]:
            price, qty = float(bid[0]), float(bid[1])
            if qty == 0 and price in st.session_state.bids:
                del st.session_state.bids[price]
            else:
                st.session_state.bids[price] = qty
        # Update asks
        for ask in data["a"]:
            price, qty = float(ask[0]), float(ask[1])
            if qty == 0 and price in st.session_state.asks:
                del st.session_state.asks[price]
            else:
                st.session_state.asks[price] = qty
    elif data.get("e") == "trade":
        st.session_state.latest_price = float(data["p"])
        st.session_state.connected = True

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://stream.binance.us:9443/ws/btcusdt@trade",
        on_message=on_message
    )
    ws.run_forever()

# Start WebSocket thread
if "ws_thread" not in st.session_state:
    ws_thread = threading.Thread(target=start_ws, daemon=True)
    ws_thread.start()
    st.session_state.ws_thread = ws_thread

# Utility function
def calculate_strength(book):
    sorted_book = dict(sorted(book.items(), reverse=True))
    top_qty = sum(list(sorted_book.values())[:5])
    total_qty = sum(sorted_book.values())
    return top_qty, total_qty

# Auto-refresh every second
st_autorefresh = st.experimental_rerun  # Use rerun for auto-update
st.write("Dashboard updating automatically every second...")

# Display
price = st.session_state.latest_price
bids = st.session_state.bids
asks = st.session_state.asks

if not st.session_state.connected:
    price_placeholder.text("Connecting to Binance US WebSocket…")
    buy_strength_placeholder.text("Buyers: Loading…")
    sell_strength_placeholder.text("Sellers: Loading…")
    order_book_placeholder.text("Order Book: Loading…")
    volume_placeholder.text("Volume: Loading…")
else:
    # Price
    price_placeholder.markdown(f"### Price: ${price:,.2f}")

    # Strength
    top_buy, total_buy = calculate_strength(bids)
    top_sell, total_sell = calculate_strength(asks)
    buy_strength_placeholder.markdown(f"**Buy Strength:** Top5 {top_buy:.2f} | Total {total_buy:.2f}")
    sell_strength_placeholder.markdown(f"**Sell Strength:** Top5 {top_sell:.2f} | Total {total_sell:.2f}")

    # Order Book
    order_book_placeholder.markdown(
        f"**Top 5 Bids:**\n{pd.DataFrame(sorted(bids.items(), reverse=True)[:5], columns=['Price','Qty']).to_markdown()}\n\n"
        f"**Top 5 Asks:**\n{pd.DataFrame(sorted(asks.items())[:5], columns=['Price','Qty']).to_markdown()}"
    )

    # Volume pushing
    buy_push = top_buy / (top_buy + top_sell) * 100 if (top_buy + top_sell) > 0 else 0
    sell_push = top_sell / (top_buy + top_sell) * 100 if (top_buy + top_sell) > 0 else 0
    volume_placeholder.markdown(f"**Buy Pressure:** {buy_push:.1f}% | **Sell Pressure:** {sell_push:.1f}%")
