import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 0.5 second (500 ms)
st_autorefresh(interval=500, key="refresh_dashboard")

st.set_page_config(page_title="BTC/USDT Buyer-Seller Dashboard", layout="wide")
st.title("💎 BTC/USDT Buyer-Seller Dashboard (Binance US)")

# Binance US order book endpoint
BASE_URL = "https://api.binance.us/api/v3/depth"
SYMBOL = "BTCUSDT"
LIMIT = 5  # Top 5 levels

# Fetch order book data
try:
    response = requests.get(f"{BASE_URL}?symbol={SYMBOL}&limit={LIMIT}")
    data = response.json()

    bids = data['bids']  # [price, qty]
    asks = data['asks']

    # Convert to DataFrame
    bid_df = pd.DataFrame(bids, columns=['Bid Price', 'Bid Qty']).astype(float)
    ask_df = pd.DataFrame(asks, columns=['Ask Price', 'Ask Qty']).astype(float)

    # Current price: mid-price between best bid and ask
    price = (bid_df['Bid Price'].iloc[0] + ask_df['Ask Price'].iloc[0]) / 2

    # Buyers & sellers strength (sum of top 5 qty)
    buyers_strength = bid_df['Bid Qty'].sum()
    sellers_strength = ask_df['Ask Qty'].sum()
    total_volume = buyers_strength + sellers_strength

    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Price (USDT)", f"${price:,.2f}")
    col2.metric("Total Volume (BTC)", f"{total_volume:.4f}")

    st.metric("Buyers Strength (BTC)", f"{buyers_strength:.4f}")
    st.metric("Sellers Strength (BTC)", f"{sellers_strength:.4f}")

    # Display top 5 order book levels
    st.subheader("Top 5 Order Book Levels")
    order_book_df = pd.concat([bid_df.reset_index(drop=True), ask_df.reset_index(drop=True)], axis=1)
    st.dataframe(order_book_df, use_container_width=True)

except Exception as e:
    st.error(f"Failed to fetch order book data: {e}")
