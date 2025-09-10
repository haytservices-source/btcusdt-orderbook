import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import collections

# --- Auto-refresh every 3 seconds ---
st_autorefresh(interval=3000, key="refresh")

st.set_page_config(page_title="BTC/USDT Advanced Order Flow", layout="wide")
st.title("📊 BTC/USDT Advanced Order Flow Dashboard")

# --- Rolling WPI memory ---
if "wpi_history" not in st.session_state:
    st.session_state.wpi_history = collections.deque(maxlen=5)

# --- Fetch Order Book ---
def get_orderbook(limit=200):
    url = "https://api.binance.com/api/v3/depth"
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

# --- Fetch Candles ---
def get_candles(limit=5):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        data = res.json()
        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume",
            "c","q","n","taker_base","taker_quote","ignore"
        ], dtype=float)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df[["time","open","high","low","close","volume"]]
    except Exception:
        return None

# --- Get Data ---
bids, asks = get_orderbook()
candles = get_candles()

if bids is None or asks is None or candles is None:
    st.error("❌ Failed to fetch market data.")
    st.stop()

# --- Mid Price ---
mid_price = (bids["price"].max() + asks["price"].min()) / 2

# --- Weighted Pressure ---
bids["weight"] = bids.apply(lambda row: row["qty"] / max(1, mid_price - row["price"]), axis=1)
asks["weight"] = asks.apply(lambda row: row["qty"] / max(1, row["price"] - mid_price), axis=1)

weighted_bids = bids["weight"].sum()
weighted_asks = asks["weight"].sum()

wpi = (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks) if weighted_bids + weighted_asks > 0 else 0

# --- Smoothed WPI ---
st.session_state.wpi_history.append(wpi)
smoothed_wpi = sum(st.session_state.wpi_history) / len(st.session_state.wpi_history)

# --- Whale Detection ---
big_bid = bids.loc[bids["qty"].idxmax()]
big_ask = asks.loc[asks["qty"].idxmax()]
nearest_bid = bids[bids["price"] < mid_price].sort_values("price", ascending=False).head(1)
nearest_ask = asks[asks["price"] > mid_price].sort_values("price", ascending=True).head(1)

nearest_bid_price, nearest_bid_qty = nearest_bid.iloc[0]["price"], nearest_bid.iloc[0]["qty"]
nearest_ask_price, nearest_ask_qty = nearest_ask.iloc[0]["price"], nearest_ask.iloc[0]["qty"]

# --- Whale Wall Strength (relative to median order size) ---
median_bid = bids["qty"].median()
median_ask = asks["qty"].median()
strong_buy_wall = nearest_bid_qty > 2 * median_bid
strong_sell_wall = nearest_ask_qty > 2 * median_ask

# --- Candle Confirmation ---
last_close = candles["close"].iloc[-1]
prev_close = candles["close"].iloc[-2]
candle_trend = "bullish" if last_close > prev_close else "bearish"

# --- Trading Logic ---
signal = "⚖️ Neutral / Wait"
if smoothed_wpi > 0.35 and strong_buy_wall and candle_trend == "bullish":
    signal = "✅ BUY Signal"
elif smoothed_wpi < -0.35 and strong_sell_wall and candle_trend == "bearish":
    signal = "❌ SELL Signal"

# --- Show Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Mid Price", f"${mid_price:,.2f}")
col2.metric("Smoothed Buy Pressure", f"{weighted_bids:,.2f}")
col3.metric("Smoothed Sell Pressure", f"{weighted_asks:,.2f}")

st.subheader(f"Market Signal → {signal}")
st.caption(f"Bias Strength (Smoothed WPI): {smoothed_wpi:.2f}")

st.caption(f"🐋 Biggest Buy Wall: {big_bid['qty']:.2f} BTC @ ${big_bid['price']:.0f} | "
           f"🐋 Biggest Sell Wall: {big_ask['qty']:.2f} BTC @ ${big_ask['price']:.0f}")

# --- Layout with two charts ---
col_left, col_right = st.columns(2)

# --- Heatmap Chart ---
with col_left:
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=bids["price"], y=bids["qty"],
                          name="Bids", marker=dict(color="green"), opacity=0.6))
    fig1.add_trace(go.Bar(x=asks["price"], y=asks["qty"],
                          name="Asks", marker=dict(color="red"), opacity=0.6))
    fig1.update_layout(
        title="Liquidity Heatmap (Order Book Depth)",
        xaxis_title="Price", yaxis_title="Quantity",
        barmode="overlay", height=500
    )
    st.plotly_chart(fig1, use_container_width=True)

# --- Candlestick Chart ---
with col_right:
    fig2 = go.Figure(data=[go.Candlestick(
        x=candles["time"],
        open=candles["open"], high=candles["high"],
        low=candles["low"], close=candles["close"],
        name="Price"
    )])
    fig2.update_layout(
        title="BTC/USDT 1m Candles",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)
