import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# Page settings
st.set_page_config(page_title="BTC/USDT Futures Live Order Book", layout="wide")
st.title("BTC/USDT (USDⓈ-M Futures) Live Order Book Dashboard")

# Auto-refresh every 0.5 seconds
st_autorefresh(interval=500, key="refresh")

# Symbol for Binance USDⓈ-M Futures BTC/USDT perpetual
SYMBOL = "BTCUSDT_PERP"

# API URLs
ORDER_BOOK_URL = f"https://fapi.binance.com/fapi/v1/depth?symbol={SYMBOL}&limit=5"
PRICE_URL = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={SYMBOL}"

def fetch_data():
    # Fetch last traded price
    try:
        price_resp = requests.get(PRICE_URL, timeout=1)
        price_resp.raise_for_status()
        price = float(price_resp.json().get('price', 0))
    except:
        price = 0.0

    # Fetch order book (top 5)
    try:
        ob_resp = requests.get(ORDER_BOOK_URL, timeout=1)
        ob_resp.raise_for_status()
        ob_data = ob_resp.json()
        bids = ob_data.get('bids', [])  # list of [price, qty]
        asks = ob_data.get('asks', [])
    except:
        bids, asks = [], []

    # Calculate Buyers/Sellers Strength and Total Volume
    buyers_strength = sum([float(bid[1]) for bid in bids])
    sellers_strength = sum([float(ask[1]) for ask in asks])
    total_volume = buyers_strength + sellers_strength

    # Prepare order book DataFrame
    df_bids = pd.DataFrame(bids, columns=['Bid Price', 'Bid Qty'])
    df_asks = pd.DataFrame(asks, columns=['Ask Price', 'Ask Qty'])
    df_orderbook = pd.concat([df_bids, df_asks], axis=1)

    return price, buyers_strength, sellers_strength, total_volume, df_orderbook

# Fetch data
price, buyers_strength, sellers_strength, total_volume, df_orderbook = fetch_data()

# Display
st.markdown(f"**Price:** ${price:,.2f}")
st.markdown(f"**Buyers Strength:** {buyers_strength:.4f} BTC")
st.markdown(f"**Sellers Strength:** {sellers_strength:.4f} BTC")
st.markdown(f"**Total Volume (Top 5):** {total_volume:.4f} BTC")
st.table(df_orderbook)

