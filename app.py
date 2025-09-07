import streamlit as st
import requests
import pandas as pd
import time

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="BTC/USD Futures Live Dashboard", layout="wide")
st.title("BTC/USD (USDⓈ-M Futures) Live Order Book Dashboard")

# -------------------------------
# Binance Futures Symbol
# -------------------------------
SYMBOL = "BTCUSD_PERP"

# URLs
ORDER_BOOK_URL = f"https://fapi.binance.com/fapi/v1/depth?symbol={SYMBOL}&limit=5"
PRICE_URL = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={SYMBOL}"

# -------------------------------
# Placeholders for dynamic updating
# -------------------------------
price_placeholder = st.empty()
buyers_placeholder = st.empty()
sellers_placeholder = st.empty()
volume_placeholder = st.empty()
orderbook_placeholder = st.empty()

# -------------------------------
# Function to fetch data
# -------------------------------
def fetch_data():
    # Fetch last traded price
    try:
        price_resp = requests.get(PRICE_URL, timeout=1)
        price_resp.raise_for_status()
        price = float(price_resp.json()['price'])
    except:
        price = 0.0

    # Fetch order book (top 5)
    try:
        ob_resp = requests.get(ORDER_BOOK_URL, timeout=1)
        ob_resp.raise_for_status()
        ob_data = ob_resp.json()
        bids = ob_data['bids']  # list of [price, qty]
        asks = ob_data['asks']
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

# -------------------------------
# Dashboard Auto-Refresh
# -------------------------------
while True:
    price, buyers_strength, sellers_strength, total_volume, df_orderbook = fetch_data()

    # Update dashboard
    price_placeholder.markdown(f"**Price:** ${price:,.2f}")
    buyers_placeholder.markdown(f"**Buyers Strength:** {buyers_strength:.4f} BTC")
    sellers_placeholder.markdown(f"**Sellers Strength:** {sellers_strength:.4f} BTC")
    volume_placeholder.markdown(f"**Total Volume (Top 5):** {total_volume:.4f} BTC")
    orderbook_placeholder.table(df_orderbook)

    # Refresh every 0.5 seconds
    time.sleep(0.5)
