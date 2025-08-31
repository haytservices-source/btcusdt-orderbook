import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="BTCUSDT Order Book", layout="wide")

# Alternative Binance API endpoint (uses data proxy)
BASE_URL = "https://api.binance.us/api/v3/depth"

def get_orderbook(symbol="BTCUSDT", limit=30):
    url = f"{BASE_URL}?symbol={symbol}&limit={limit}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

st.title("📊 BTC/USDT Live Order Book")

levels = st.slider("Order book levels per side", 5, 100, 30, 5)

try:
    data = get_orderbook("BTCUSDT", levels)

    bids = pd.DataFrame(data["bids"], columns=["Price", "Amount"], dtype=float)
    asks = pd.DataFrame(data["asks"], columns=["Price", "Amount"], dtype=float)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bids")
        st.dataframe(bids)

    with col2:
        st.subheader("Asks")
        st.dataframe(asks)

except Exception as e:
    st.error(f"Failed to load order book: {e}")
