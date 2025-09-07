import streamlit as st
import websocket
import json
import threading

st.title("BTC/USDT (USDⓈ-M Futures) Live Price")

price_placeholder = st.empty()

def on_message(ws, message):
    data = json.loads(message)
    price = float(data['p'])
    price_placeholder.metric(label="BTC/USDT Price", value=f"${price:,.2f}")

def run_ws():
    ws = websocket.WebSocketApp(
        "wss://fstream.binance.com/ws/btcusdt@trade",
        on_message=on_message
    )
    ws.run_forever()

thread = threading.Thread(target=run_ws)
thread.start()
