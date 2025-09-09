import streamlit as st
import requests
import pandas as pd
import time
import plotly.graph_objects as go

st.set_page_config(page_title="BTC/USDT Order Book Dashboard", layout="wide")
st.title("BTC/USDT Live Order Book (Binance US)")

# ------------------ Placeholders ------------------
price_placeholder = st.empty()
agg_placeholder = st.empty()
bids_placeholder = st.empty()
asks_placeholder = st.empty()
chart_placeholder = st.empty()

# ------------------ Binance US Order Book API ------------------
BINANCE_OB_URL = "https://api.binance.us/api/v3/depth?symbol=BTCUSDT&limit=20"

def get_order_book():
    try:
        response = requests.get(BINANCE_OB_URL, timeout=3)
        response.raise_for_status()
        data = response.json()
        bids = pd.DataFrame(data['bids'], columns=['Price', 'Quantity']).astype(float)
        asks = pd.DataFrame(data['asks'], columns=['Price', 'Quantity']).astype(float)
        return bids, asks
    except Exception:
        return None, None

def get_mid_price(bids, asks):
    return (bids['Price'].iloc[0] + asks['Price'].iloc[0]) / 2

# ------------------ Keep history for chart ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ Draw static headers once ------------------
bids_placeholder.subheader("Top 10 Bids")
asks_placeholder.subheader("Top 10 Asks")

# ------------------ Live update loop ------------------
while True:
    bids, asks = get_order_book()

    if bids is not None and asks is not None:
        # Mid price
        mid_price = get_mid_price(bids, asks)
        price_placeholder.metric(label="Mid Price", value=f"${mid_price:,.2f}")

        # Aggressiveness / sentiment
        total_bid_qty = bids['Quantity'].sum()
        total_ask_qty = asks['Quantity'].sum()

        if total_bid_qty > total_ask_qty:
            market_sentiment = "BUYERS AGGRESSIVE 🟢"
        elif total_ask_qty > total_bid_qty:
            market_sentiment = "SELLERS AGGRESSIVE 🔴"
        else:
            market_sentiment = "NEUTRAL ⚪"

        agg_placeholder.markdown(
            f"**Market Sentiment:** {market_sentiment}<br>"
            f"**Total Bid Volume:** {total_bid_qty:,.4f} BTC | "
            f"**Total Ask Volume:** {total_ask_qty:,.4f} BTC",
            unsafe_allow_html=True
        )

        # Update tables
        bids_placeholder.dataframe(
            bids.head(10).style.format({"Price": "${:,.2f}", "Quantity": "{:,.4f}"}),
            use_container_width=True
        )
        asks_placeholder.dataframe(
            asks.head(10).style.format({"Price": "${:,.2f}", "Quantity": "{:,.4f}"}),
            use_container_width=True
        )

        # ------------------ Update history & chart ------------------
        st.session_state.history.append(
            {"price": mid_price, "bids": total_bid_qty, "asks": total_ask_qty}
        )

        # Keep last 30 points only
        st.session_state.history = st.session_state.history[-30:]

        df_hist = pd.DataFrame(st.session_state.history)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df_hist["bids"], mode="lines+markers", name="Buy Volume", line=dict(color="green")
        ))
        fig.add_trace(go.Scatter(
            y=df_hist["asks"], mode="lines+markers", name="Sell Volume", line_
