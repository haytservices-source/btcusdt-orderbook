import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# Page setup
st.set_page_config(page_title="BTC/USDT Futures Order Book", layout="wide")
st.title("BTC/USDT (USDⓈ-M Futures) Live Order Book Dashboard")

# Auto-refresh every 0.5 seconds
st_autorefresh(interval=500, key="refresh")

SYMBOL = "BTCUSDT"
PRICE_URL = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={SYMBOL}"
ORDER_BOOK_URL = f"https://fapi.binance.com/fapi/v1/depth?symbol={SYMBOL}&limit=5"

def fetch_data():
    try:
        # Fetch last price
        price_data = requests.get(PRICE_URL, timeout=1).json()
        price = float(price_data.get('price', 0))

        # Fetch order book
        ob_data = requests.get(ORDER_BOOK_URL, timeout=1).json()
        bids = ob_data.get('bids', [])
        asks = ob_data.get('asks', [])

        buyers_strength = sum(float(bid[1]) for bid in bids)
        sellers_strength = sum(float(ask[1]) for ask in asks)
        total_volume = buyers_strength + sellers_strength

        df_bids = pd.DataFrame(bids, columns=['Bid Price', 'Bid Qty'])
        df_asks = pd.DataFrame(asks, columns=['Ask Price', 'Ask Qty'])
        df_orderbook = pd.concat([df_bids, df_asks], axis=1)
    except Exception as e:
        price = buyers_strength = sellers_strength = total_volume = 0
        df_orderbook = pd.DataFrame(columns=['Bid Price', 'Bid Qty', 'Ask Price', 'Ask Qty'])
        st.error(f"Error fetching data: {e}")

    return price, buyers_strength, sellers_strength, total_volume, df_orderbook

price, buyers_strength, sellers_strength, total_volume, df_orderbook = fetch_data()

st.markdown(f"**Price:** ${price:,.2f}")
st.markdown(f"**Buyers Strength:** {buyers_strength:.4f} BTC")
st.markdown(f"**Sellers Strength:** {sellers_strength:.4f} BTC")
st.markdown(f"**Total Volume (Top 5):** {total_volume:.4f} BTC")
st.table(df_orderbook)
