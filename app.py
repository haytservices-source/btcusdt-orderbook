import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Binance API
BASE_URL = "https://api.binance.com/api/v3/depth"

def get_orderbook(symbol="BTCUSDT", limit=20):
    url = f"{BASE_URL}?symbol={symbol}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    bids = pd.DataFrame(data["bids"], columns=["Price", "Quantity"], dtype=float)
    asks = pd.DataFrame(data["asks"], columns=["Price", "Quantity"], dtype=float)
    return bids, asks

st.set_page_config(page_title="BTC/USDT Order Book", layout="wide")

# Auto-refresh every 1 second
st_autorefresh(interval=1000, key="orderbook_refresh")

st.title("📊 BTC/USDT Live Order Book (1s Refresh)")

bids, asks = get_orderbook()

# Plot order book depth
fig = go.Figure()
fig.add_trace(go.Bar(x=bids["Price"], y=bids["Quantity"], name="Bids", marker_color="green"))
fig.add_trace(go.Bar(x=asks["Price"], y=asks["Quantity"], name="Asks", marker_color="red"))

fig.update_layout(title="Order Book Depth", xaxis_title="Price", yaxis_title="Quantity", barmode="overlay")

st.plotly_chart(fig, use_container_width=True)

# Quick Buy/Sell Pressure
total_bids = bids["Quantity"].sum()
total_asks = asks["Quantity"].sum()

if total_bids > total_asks:
    st.success("📈 Buyers stronger right now → Possible Up Move")
elif total_bids < total_asks:
    st.error("📉 Sellers stronger right now → Possible Down Move")
else:
    st.warning("➖ Balanced → Sideways")
