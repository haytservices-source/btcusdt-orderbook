import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every 2 seconds ---
st_autorefresh(interval=2000, key="refresh")

st.set_page_config(page_title="BTC/USDT Order Book Dashboard", layout="wide")
st.title("📊 BTC/USDT Order Book (Binance US)")

# --- Get Order Book Data ---
def get_orderbook():
    url = "https://api.binance.us/api/v3/depth"
    params = {"symbol": "BTCUSDT", "limit": 50}
    try:
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        data = res.json()
        bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)
        return bids, asks
    except Exception:
        return None, None

bids, asks = get_orderbook()

if bids is None or asks is None:
    st.error("❌ Failed to fetch order book.")
    st.stop()

# --- Calculate Buy/Sell Pressure ---
total_bids = bids["qty"].sum()
total_asks = asks["qty"].sum()

if total_bids + total_asks > 0:
    imbalance = (total_bids - total_asks) / (total_bids + total_asks)
else:
    imbalance = 0

if imbalance > 0.2:
    pressure_text = "🔥 Buyers Aggressive (Bullish)"
elif imbalance < -0.2:
    pressure_text = "🔴 Sellers Aggressive (Bearish)"
else:
    pressure_text = "⚖️ Balanced (Sideways)"

# --- Show Live Pressure ---
col1, col2, col3 = st.columns(3)
col1.metric("Buy Volume", f"{total_bids:,.2f}")
col2.metric("Sell Volume", f"{total_asks:,.2f}")
col3.metric("Imbalance", f"{imbalance*100:.1f}%")
st.subheader(f"Market Pressure: {pressure_text}")

# --- Heatmap Zones (Top 5 Walls) ---
top_bid = bids.sort_values("qty", ascending=False).head(5)
top_ask = asks.sort_values("qty", ascending=False).head(5)

st.subheader("Liquidity Heatmap Zones")
heatmap_df = pd.concat([
    top_bid.assign(side="Buy Wall"),
    top_ask.assign(side="Sell Wall")
])
st.table(heatmap_df)

# --- Plot Heatmap ---
fig = go.Figure()

fig.add_trace(go.Bar(
    x=bids["price"], y=bids["qty"],
    name="Bids", marker_color="green", opacity=0.6
))
fig.add_trace(go.Bar(
    x=asks["price"], y=asks["qty"],
    name="Asks", marker_color="red", opacity=0.6
))

fig.update_layout(
    title="Order Book Heatmap (Top 50 Levels)",
    xaxis_title="Price",
    yaxis_title="Quantity",
    barmode="overlay",
    height=500
)

st.plotly_chart(fig, use_container_width=True)
