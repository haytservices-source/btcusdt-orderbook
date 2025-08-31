import time
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------- Page config ----------
st.set_page_config(page_title="BTCUSDT Order Book", layout="wide", page_icon="📊")
st.title("📊 BTC/USDT Live Order Book")

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
depth = st.sidebar.slider("Order book levels per side", 10, 100, 30, step=10)
auto_refresh_secs = st.sidebar.slider("Auto-refresh (seconds)", 0, 30, 5)
if auto_refresh_secs > 0:
    st_autorefresh(interval=auto_refresh_secs * 1000, key="autorefresh")

symbol = "BTCUSDT"

# ---------- Data fetch ----------
@st.cache_data(ttl=5, show_spinner=False)
def get_orderbook(symbol: str, limit: int):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    bids = pd.DataFrame(data["bids"], columns=["price", "amount"], dtype=float)
    asks = pd.DataFrame(data["asks"], columns=["price", "amount"], dtype=float)
    bids["total_usdt"] = bids["price"] * bids["amount"]
    asks["total_usdt"] = asks["price"] * asks["amount"]
    return bids, asks

try:
    bids, asks = get_orderbook(symbol, depth)
except Exception as e:
    st.error(f"Failed to load order book: {e}")
    st.stop()

# ---------- Top-of-book & metrics ----------
best_bid = bids["price"].max()
best_ask = asks["price"].min()
spread = best_ask - best_bid
mid = (best_bid + best_ask) / 2

buy_btc = bids["amount"].sum()
sell_btc = asks["amount"].sum()
buy_usd = bids["total_usdt"].sum()
sell_usd = asks["total_usdt"].sum()
total_usd = buy_usd + sell_usd
imbalance = (buy_usd / total_usd) if total_usd > 0 else 0.5

m1, m2, m3, m4 = st.columns(4)
m1.metric("Best Bid", f"{best_bid:,.2f} USDT")
m2.metric("Best Ask", f"{best_ask:,.2f} USDT")
m3.metric("Spread", f"{spread:,.2f} USDT")
m4.metric("Mid Price", f"{mid:,.2f} USDT")

c1, c2, c3 = st.columns(3)
c1.metric("Buy Depth (BTC)", f"{buy_btc:,.3f}")
c2.metric("Sell Depth (BTC)", f"{sell_btc:,.3f}")
c3.metric("Buy Share ($)", f"{imbalance*100:,.1f}%")

# Simple recommendation based on depth by dollars
if imbalance >= 0.55:
    st.success("🟢 Buy pressure dominant (≥55% of visible depth).")
elif imbalance <= 0.45:
    st.error("🔴 Sell pressure dominant (≤45% buy share).")
else:
    st.warning("⚖️ Balanced order book — consider waiting for a break.")

# ---------- Biggest walls ----------
left, right = st.columns(2)
with left:
    st.subheader("Top Bid Walls (Support)")
    st.dataframe(
        bids.nlargest(5, "total_usdt")[["price", "amount", "total_usdt"]]
            .rename(columns={"price": "Price", "amount": "BTC", "total_usdt": "USDT"}),
        use_container_width=True,
    )
with right:
    st.subheader("Top Ask Walls (Resistance)")
    st.dataframe(
        asks.nlargest(5, "total_usdt")[["price", "amount", "total_usdt"]]
            .rename(columns={"price": "Price", "amount": "BTC", "total_usdt": "USDT"}),
        use_container_width=True,
    )

# ---------- Depth chart ----------
st.subheader("Order Book Depth")
bids_sorted = bids.sort_values("price")
asks_sorted = asks.sort_values("price")

fig = go.Figure()
fig.add_trace(go.Bar(x=bids_sorted["price"], y=bids_sorted["amount"], name="Bids"))
fig.add_trace(go.Bar(x=asks_sorted["price"], y=asks_sorted["amount"], name="Asks"))
fig.update_layout(
    xaxis_title="Price (USDT)",
    yaxis_title="Amount (BTC)",
    legend_title_text="Side",
    bargap=0.05,
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Note: This reads the visible top-of-book depth from Binance. Walls can move/cancel quickly; "
    "treat as short-term context, not financial advice."
)
