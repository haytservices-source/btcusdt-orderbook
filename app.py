import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go

st.set_page_config(page_title="BTC/USDT Order Book Dashboard", layout="wide")
st.title("📊 BTC/USDT Live Order Book (Binance US)")

# API fetch function
def get_orderbook(symbol="BTCUSDT", limit=50):
    url = f"https://api.binance.us/api/v3/depth?symbol={symbol}&limit={limit}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        bids = [(float(p), float(q)) for p, q in data["bids"]]
        asks = [(float(p), float(q)) for p, q in data["asks"]]
        return bids, asks
    except Exception as e:
        st.error(f"Error fetching order book: {e}")
        return [], []

# Pressure calculation
def analyze_pressure(bids, asks):
    total_bid_vol = sum(q for _, q in bids)
    total_ask_vol = sum(q for _, q in asks)

    imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-9)

    if imbalance > 0.2:
        bias = "🔥 Buyers Strong (Bullish)"
    elif imbalance < -0.2:
        bias = "💀 Sellers Strong (Bearish)"
    else:
        bias = "➖ Neutral / Sideways"

    return total_bid_vol, total_ask_vol, imbalance, bias

# Whale wall detection
def detect_walls(levels, top_n=3):
    """Find top N largest liquidity walls"""
    df = pd.DataFrame(levels, columns=["price", "volume"])
    df["score"] = df["volume"] / df["volume"].sum()
    df = df.sort_values("volume", ascending=False).head(top_n)
    return df

# Live dashboard
refresh_rate = 2  # seconds
placeholder = st.empty()

while True:
    bids, asks = get_orderbook(limit=50)

    if not bids or not asks:
        time.sleep(refresh_rate)
        continue

    # Pressure analysis
    total_bid, total_ask, imbalance, bias = analyze_pressure(bids, asks)

    # Walls
    bid_walls = detect_walls(bids, top_n=3)
    ask_walls = detect_walls(asks, top_n=3)

    # Plot order book depth chart
    bid_prices, bid_vols = zip(*bids)
    ask_prices, ask_vols = zip(*asks)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=bid_prices, y=bid_vols, name="Bids", marker_color="green"))
    fig.add_trace(go.Bar(x=ask_prices, y=ask_vols, name="Asks", marker_color="red"))

    fig.update_layout(
        title="Order Book Heatmap",
        xaxis_title="Price",
        yaxis_title="Volume",
        template="plotly_dark",
        barmode="overlay",
        height=500
    )

    with placeholder.container():
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Buy Volume", f"{total_bid:,.2f}")
            st.metric("Total Sell Volume", f"{total_ask:,.2f}")
            st.metric("Imbalance", f"{imbalance*100:.1f}%")
            st.subheader(f"📈 Market Bias: {bias}")

        with col2:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🛑 Major Liquidity Walls")
        c1, c2 = st.columns(2)

        with c1:
            st.write("**Top Buy Walls (Support):**")
            st.dataframe(bid_walls)

        with c2:
            st.write("**Top Sell Walls (Resistance):**")
            st.dataframe(ask_walls)

    time.sleep(refresh_rate)
