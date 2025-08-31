# app.py
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# ---------- Config ----------
st.set_page_config(page_title="BTC/USDT Order Book (5s)", layout="wide", page_icon="📊")
st_autorefresh(interval=5000, key="orderbook_autorefresh")  # refresh every 5000 ms = 5 sec

BASE_URL = "https://api.binance.us/api/v3/depth"  # the endpoint that worked for you
SYMBOL = "BTCUSDT"

st.title("📊 BTC/USDT Live Order Book (5s refresh)")
st.caption("Data from api.binance.us · Not financial advice")

# ---------- Controls ----------
with st.sidebar:
    st.header("Settings")
    levels = st.slider("Order book levels per side", min_value=10, max_value=100, value=30, step=10)
    show_chart = st.checkbox("Show depth chart", value=True)

# ---------- Fetch function ----------
@st.cache_data(ttl=4)  # cache briefly so repeated calls inside same second don't re-fetch
def fetch_orderbook(symbol: str, limit: int):
    url = f"{BASE_URL}?symbol={symbol}&limit={limit}"
    resp = requests.get(url, timeout=6)
    resp.raise_for_status()
    data = resp.json()
    if "bids" not in data or "asks" not in data:
        raise ValueError("Invalid response from Binance (no bids/asks).")
    bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
    asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)
    bids["total_usd"] = bids["price"] * bids["qty"]
    asks["total_usd"] = asks["price"] * asks["qty"]
    return bids, asks

# ---------- Load data ----------
try:
    bids, asks = fetch_orderbook(SYMBOL, levels)
    last_ok = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
except Exception as e:
    st.error(f"Failed to load order book: {e}")
    st.info("The app will auto-retry every 5 seconds.")
    st.stop()

# ---------- Metrics & simple dashboard ----------
best_bid = bids["price"].max()
best_ask = asks["price"].min()
spread = best_ask - best_bid
mid = (best_bid + best_ask) / 2

buy_btc = bids["qty"].sum()
sell_btc = asks["qty"].sum()
buy_usd = bids["total_usd"].sum()
sell_usd = asks["total_usd"].sum()
total_usd = buy_usd + sell_usd if (buy_usd + sell_usd) > 0 else 1
buy_share_pct = buy_usd / total_usd

# Top info line
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Bid", f"{best_bid:,.2f} USDT")
c2.metric("Best Ask", f"{best_ask:,.2f} USDT")
c3.metric("Spread", f"{spread:,.2f} USDT")
c4.metric("Last update", last_ok)

# Depth & decision
st.subheader("Order Book Snapshot")
left, right = st.columns([1, 1])

with left:
    st.markdown("**Top Bid Walls (Support)**")
    top_bids = bids.nlargest(8, "total_usd")[["price", "qty", "total_usd"]].rename(
        columns={"price": "Price", "qty": "BTC", "total_usd": "USDT"}
    )
    st.dataframe(top_bids, use_container_width=True)

with right:
    st.markdown("**Top Ask Walls (Resistance)**")
    top_asks = asks.nlargest(8, "total_usd")[["price", "qty", "total_usd"]].rename(
        columns={"price": "Price", "qty": "BTC", "total_usd": "USDT"}
    )
    st.dataframe(top_asks, use_container_width=True)

# Quick depth metrics row
d1, d2, d3 = st.columns(3)
d1.metric("Buy Depth (BTC)", f"{buy_btc:,.3f}")
d2.metric("Sell Depth (BTC)", f"{sell_btc:,.3f}")
d3.metric("Buy Share (visible $)", f"{buy_share_pct*100:,.1f}%")

# Simple recommendation logic
if buy_share_pct >= 0.55:
    st.success("🟢 BUYERS STRONG — visible buy-side is dominant (≥55%)")
elif buy_share_pct <= 0.45:
    st.error("🔴 SELLERS STRONG — visible sell-side is dominant (≤45%)")
else:
    st.info("⚖️ BALANCED — no clear dominance (between 45% and 55%)")

# ---------- Depth chart ----------
if show_chart:
    st.subheader("Depth Chart (Amounts by Price)")
    bids_sorted = bids.sort_values("price")
    asks_sorted = asks.sort_values("price")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bids_sorted["price"], y=bids_sorted["qty"], name="Bids (buy)", marker_color="green", opacity=0.7))
    fig.add_trace(go.Bar(x=asks_sorted["price"], y=asks_sorted["qty"], name="Asks (sell)", marker_color="red", opacity=0.7))
    fig.update_layout(
        xaxis_title="Price (USDT)",
        yaxis_title="Amount (BTC)",
        legend_title_text="Side",
        bargap=0.02,
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption("Note: this reads visible top-of-book depth only. Walls can cancel or move quickly. Use as short-term context, not financial advice.")
