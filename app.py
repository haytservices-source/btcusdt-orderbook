import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(page_title="BTC/USDT Futures Live Order Book", layout="wide")
st.title("BTC/USDT (Binance USDⓈ-M Futures) Live Order Book")

SYMBOL = "BTCUSDT"
PRICE_URL = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={SYMBOL}"
ORDER_BOOK_URL = f"https://fapi.binance.com/fapi/v1/depth?symbol={SYMBOL}&limit=5"

price_placeholder = st.empty()
buyers_placeholder = st.empty()
sellers_placeholder = st.empty()
volume_placeholder = st.empty()
orderbook_placeholder = st.empty()

def fetch_data():
    try:
        price_data = requests.get(PRICE_URL, timeout=2).json()
        price = float(price_data['price'])
    except:
        price = 0.0

    try:
        ob_data = requests.get(ORDER_BOOK_URL, timeout=2).json()
        bids = ob_data.get('bids', [])
        asks = ob_data.get('asks', [])
    except:
        bids, asks = [], []

    buyers_strength = sum(float(bid[1]) for bid in bids)
    sellers_strength = sum(float(ask[1]) for ask in asks)
    total_volume = buyers_strength + sellers_strength

    df_bids = pd.DataFrame(bids, columns=['Bid Price', 'Bid Qty'])
    df_asks = pd.DataFrame(asks, columns=['Ask Price', 'Ask Qty'])
    df_orderbook = pd.concat([df_bids, df_asks], axis=1)

    return price, buyers_strength, sellers_strength, total_volume, df_orderbook

while True:
    price, buyers_strength, sellers_strength, total_volume, df_orderbook = fetch_data()

    price_placeholder.markdown(f"**Price:** ${price:,.2f}")
    buyers_placeholder.markdown(f"**Buyers Strength:** {buyers_strength:.4f} BTC")
    sellers_placeholder.markdown(f"**Sellers Strength:** {sellers_strength:.4f} BTC")
    volume_placeholder.markdown(f"**Total Volume (Top 5):** {total_volume:.4f} BTC")
    orderbook_placeholder.table(df_orderbook)

    time.sleep(1)
