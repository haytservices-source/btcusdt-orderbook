import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every 2 seconds ---
st_autorefresh(interval=2000, key="refresh")

st.set_page_config(page_title="BTC/USDT Advanced Order Book", layout="wide")
st.title("📊 BTC/USDT Advanced Order Flow Dashboard")

# --- Fetch Order Book ---
def get_orderbook(limit=200):
    url = "https://api.binance.us/api/v3/depth"
    params = {"symbol": "BTCUSDT", "limit": limit}
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

# --- Mid Price ---
mid_price = (bids["price"].max() + asks["price"].min()) / 2

# --- Weighted Pressure (distance-weighted liquidity) ---
bids["weight"] = bids.apply(lambda row: row["qty"] / max(1, mid_price - row["price"]), axis=1)
asks["weight"] = asks.apply(lambda row: row["qty"] / max(1, row["price"] - mid_price), axis=1)

weighted_bids = bids["weight"].sum()
weighted_asks = asks["weight"].sum()

if weighted_bids + weighted_asks > 0:
    wpi = (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks)
else:
    wpi = 0

# --- Bias Interpretation ---
if wpi > 0.25:
    bias = "🔥 Buyers Dominant (Bullish)"
    confidence = f"{wpi*100:.1f}%"
elif wpi < -0.25:
    bias = "🔴 Sellers Dominant (Bearish)"
    confidence = f"{abs(wpi)*100:.1f}%"
else:
    bias = "⚖️ Neutral / Sideways"
    confidence = f"{abs(wpi)*100:.1f}%"

# --- Whale Wall Detection ---
big_bid = bids.loc[bids["qty"].idxmax()]
big_ask = asks.loc[asks["qty"].idxmax()]

nearest_bid = bids[bids["price"] < mid_price].sort_values("price", ascending=False).head(1)
nearest_ask = asks[asks["price"] > mid_price].sort_values("price", ascending=True).head(1)

nearest_bid_price, nearest_bid_qty = nearest_bid.iloc[0]["price"], nearest_bid.iloc[0]["qty"]
nearest_ask_price, nearest_ask_qty = nearest_ask.iloc[0]["price"], nearest_ask.iloc[0]["qty"]

# --- Projection Logic ---
if nearest_bid_qty > nearest_ask_qty and wpi > 0:
    projection = f"📈 Likely Upward → Next Zone ${nearest_ask_price:,.0f}"
elif nearest_ask_qty > nearest_bid_qty and wpi < 0:
    projection = f"📉 Likely Downward → Next Zone ${nearest_bid_price:,.0f}"
else:
    projection = f"🤔 Unclear → Range ${nearest_bid_price:,.0f} - ${nearest_ask_price:,.0f}"

# --- Show Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Mid Price", f"${mid_price:,.2f}")
col2.metric("Weighted Buy Pressure", f"{weighted_bids:,.2f}")
col3.metric("Weighted Sell Pressure", f"{weighted_asks:,.2f}")

st.subheader(f"Market Bias → {bias} | Confidence: {confidence}")
st.subheader(f"Projection → {projection}")

st.caption(f"🐋 Biggest Buy Wall: {big_bid['qty']:.2f} BTC @ ${big_bid['price']:.0f} | "
           f"🐋 Biggest Sell Wall: {big_ask['qty']:.2f} BTC @ ${big_ask['price']:.0f}")

# --- Heatmap Chart ---
fig = go.Figure()

fig.add_trace(go.Bar(
    x=bids["price"], y=bids["qty"],
    name="Bids", marker=dict(color="green"), opacity=0.6
))
fig.add_trace(go.Bar(
    x=asks["price"], y=asks["qty"],
    name="Asks", marker=dict(color="red"), opacity=0.6
))

fig.update_layout(
    title="Liquidity Heatmap (Order Book Depth)",
    xaxis_title="Price",
    yaxis_title="Quantity",
    barmode="overlay",
    height=500
)

st.plotly_chart(fig, use_container_width=True)
