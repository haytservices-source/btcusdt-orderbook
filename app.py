import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every second
st_autorefresh(interval=1000, key="refresh_dashboard")

st.set_page_config(page_title="BTC/USDT Buyers vs Sellers Dashboard", layout="wide")

st.title("💎 BTC/USDT Buyers vs Sellers Dashboard")

# Binance US API endpoints
SYMBOL = "BTCUSDT"
BASE_URL = "https://api.binance.us/api/v3"

# Function to fetch latest price
def get_price(symbol):
    url = f"{BASE_URL}/ticker/price?symbol={symbol}"
    data = requests.get(url).json()
    return float(data['price'])

# Function to fetch order book
def get_order_book(symbol, limit=5):
    url = f"{BASE_URL}/depth?symbol={symbol}&limit=10"
    data = requests.get(url).json()
    bids = data['bids'][:limit]
    asks = data['asks'][:limit]
    return bids, asks

# Function to calculate buyer/seller strength
def calculate_strength(bids, asks):
    buyer_strength = sum([float(qty) for price, qty in bids])
    seller_strength = sum([float(qty) for price, qty in asks])
    total_volume = buyer_strength + seller_strength
    return buyer_strength, seller_strength, total_volume

try:
    # Fetch data
    price = get_price(SYMBOL)
    bids, asks = get_order_book(SYMBOL)
    buyer_strength, seller_strength, total_volume = calculate_strength(bids, asks)

    # Display live metrics
    st.subheader(f"Price: ${price:,.2f}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Buyers Strength", f"{buyer_strength:.4f} BTC")
    col2.metric("Sellers Strength", f"{seller_strength:.4f} BTC")
    col3.metric("Total Volume", f"{total_volume:.4f} BTC")

    # Display top 5 order book levels
    st.subheader("Top 5 Order Book Levels")
    order_book_data = []
    for i in range(len(bids)):
        bid_price, bid_qty = bids[i]
        ask_price, ask_qty = asks[i]
        order_book_data.append({
            "Bid Price": float(bid_price),
            "Bid Qty": float(bid_qty),
            "Ask Price": float(ask_price),
            "Ask Qty": float(ask_qty)
        })

    df_order_book = pd.DataFrame(order_book_data)
    st.dataframe(df_order_book, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching data: {e}")
