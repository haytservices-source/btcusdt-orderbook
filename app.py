import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time

# Binance API URL for order book
BASE_URL = "https://api.binance.com/api/v3/depth"

# Function to fetch order book
def get_order_book(symbol="BTCUSDT", limit=20):
    try:
        response = requests.get(BASE_URL, params={"symbol": symbol, "limit": limit}, timeout=5)
        response.raise_for_status()
        data = response.json()
        bids = pd.DataFrame(data["bids"], columns=["Price", "Quantity"], dtype=float)
        asks = pd.DataFrame(data["asks"], columns=["Price", "Quantity"], dtype=float)
        return bids, asks
    except Exception as e:
        st.error(f"Failed to load order book data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Streamlit App
st.set_page_config(page_title="BTCUSDT Order Book", layout="wide")

st.title("📊 Live BTC/USDT Order Book (Binance)")
st.caption("Updates every second")

# Container for updating
placeholder = st.empty()

while True:
    bids, asks = get_order_book("BTCUSDT", 20)

    if not bids.empty and not asks.empty:
        with placeholder.container():
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Buy Orders (Bids)")
                st.dataframe(bids, use_container_width=True)

            with col2:
                st.subheader("Sell Orders (Asks)")
                st.dataframe(asks, use_container_width=True)

            # Plot depth chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bids["Price"], y=bids["Quantity"],
                                     mode="lines", name="Bids", line=dict(color="green")))
            fig.add_trace(go.Scatter(x=asks["Price"], y=asks["Quantity"],
                                     mode="lines", name="Asks", line=dict(color="red")))
            fig.update_layout(title="Order Book Depth", xaxis_title="Price", yaxis_title="Quantity")
            st.plotly_chart(fig, use_container_width=True)

    time.sleep(1)
