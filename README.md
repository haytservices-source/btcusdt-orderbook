import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every 1 second ---
st_autorefresh(interval=1000, key="refresh")  # 1000ms = 1s

# --- Page Setup ---
st.set_page_config(page_title="BTC/USDT Advanced Order Flow", layout="wide")
st.title("📊 BTC/USDT Advanced Order Flow Dashboard")

# --- Initialize WPI history ---
if "wpi_history" not in st.session_state:
    st.session_state.wpi_history = []

# --- User Inputs ---
interval = st.sidebar.selectbox(
    "Select Candle Interval", ["1m", "5m", "15m", "1h", "4h"], index=0
)
candle_limit = st.sidebar.number_input("Number of Candles", 20, 200, 50)

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

# --- Fetch Candlestick Data ---
def get_candles(limit=50, interval="1m"):
    url = "https://api.binance.us/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
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
candles = get_candles(limit=candle_limit, interval=interval)

if bids is None or asks is None or candles is None:
    st.error("❌ Failed to fetch market data.")
    st.stop()

# --- Current Market Price ---
current_price = candles["close"].iloc[-1]

# --- Weighted Pressure Calculation ---
bid_dist = (current_price - bids["price"]).clip(lower=0.01)
ask_dist = (asks["price"] - current_price).clip(lower=0.01)

weighted_bids = (bids["qty"] / bid_dist).sum()
weighted_asks = (asks["qty"] / ask_dist).sum()

# --- Weighted Pressure Index ---
wpi = (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks)
st.session_state.wpi_history.append({"time": pd.Timestamp.now(), "wpi": wpi})
st.session_state.wpi_history = st.session_state.wpi_history[-100:]
wpi_df = pd.DataFrame(st.session_state.wpi_history)

# --- Market Bias ---
if wpi > 0.25:
    bias = "🔥 Buyers Dominant (Bullish)"
    confidence = f"{wpi*100:.1f}%"
elif wpi < -0.25:
    bias = "🔴 Sellers Dominant (Bearish)"
    confidence = f"{abs(wpi)*100:.1f}%"
else:
    bias = "⚖️ Neutral / Sideways"
    confidence = f"{abs(wpi)*100:.1f}%"

# --- Whale Detection ---
big_bid = bids.loc[bids["qty"].idxmax()]
big_ask = asks.loc[asks["qty"].idxmax()]

nearest_bid = bids[bids["price"] < current_price].sort_values("price", ascending=False).head(1)
nearest_ask = asks[asks["price"] > current_price].sort_values("price", ascending=True).head(1)

nearest_bid_price, nearest_bid_qty = nearest_bid.iloc[0]["price"], nearest_bid.iloc[0]["qty"]
nearest_ask_price, nearest_ask_qty = nearest_ask.iloc[0]["price"], nearest_ask.iloc[0]["qty"]

# --- Projection ---
if nearest_bid_qty > nearest_ask_qty and wpi > 0:
    projection = f"📈 Likely Upward → Next Zone ${nearest_ask_price:,.0f}"
elif nearest_ask_qty > nearest_bid_qty and wpi < 0:
    projection = f"📉 Likely Downward → Next Zone ${nearest_bid_price:,.0f}"
else:
    projection = f"🤔 Unclear → Range ${nearest_bid_price:,.0f} - ${nearest_ask_price:,.0f}"

# --- Show Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric("Weighted Buy Pressure", f"{weighted_bids:,.2f}")
col3.metric("Weighted Sell Pressure", f"{weighted_asks:,.2f}")

st.subheader(f"Market Bias → {bias} | Confidence: {confidence}")
st.subheader(f"Projection → {projection}")
st.caption(f"🐋 Biggest Buy Wall: {big_bid['qty']:.2f} BTC @ ${big_bid['price']:.0f} | "
           f"🐋 Biggest Sell Wall: {big_ask['qty']:.2f} BTC @ ${big_ask['price']:.0f}")

# --- Layout with 3 Charts ---
col_left, col_center, col_right = st.columns([1, 1, 1])

# --- Heatmap (Cumulative Depth) ---
with col_left:
    bids["cum_qty"] = bids["qty"].cumsum()
    asks["cum_qty"] = asks["qty"].cumsum()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=bids["price"], y=bids["cum_qty"], 
                              mode="lines", name="Bid Depth", line=dict(color="green")))
    fig1.add_trace(go.Scatter(x=asks["price"], y=asks["cum_qty"], 
                              mode="lines", name="Ask Depth", line=dict(color="red")))
    fig1.update_layout(
        title="Liquidity Depth (Order Book)",
        xaxis_title="Price", yaxis_title="Cumulative Quantity",
        height=500
    )
    st.plotly_chart(fig1, use_container_width=True)

# --- WPI Line Chart with Color Zones ---
with col_center:
    fig_wpi = go.Figure()
    
    bullish = wpi_df[wpi_df["wpi"] > 0.25]
    bearish = wpi_df[wpi_df["wpi"] < -0.25]
    neutral = wpi_df[(wpi_df["wpi"] >= -0.25) & (wpi_df["wpi"] <= 0.25)]
    
    if not bullish.empty:
        fig_wpi.add_trace(go.Scatter(
            x=bullish["time"], y=bullish["wpi"], mode="lines+markers",
            name="Bullish WPI", line=dict(color="green")
        ))
    if not neutral.empty:
        fig_wpi.add_trace(go.Scatter(
            x=neutral["time"], y=neutral["wpi"], mode="lines+markers",
            name="Neutral WPI", line=dict(color="gold")
        ))
    if not bearish.empty:
        fig_wpi.add_trace(go.Scatter(
            x=bearish["time"], y=bearish["wpi"], mode="lines+markers",
            name="Bearish WPI", line=dict(color="red")
        ))
    
    fig_wpi.update_layout(
        title="Live Weighted Pressure Index (WPI)",
        xaxis_title="Time", yaxis_title="WPI",
        yaxis=dict(range=[-1,1]),
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig_wpi, use_container_width=True)

# --- Candlestick Chart with Volume ---
with col_right:
    fig2 = go.Figure()
    # Candles
    fig2.add_trace(go.Candlestick(
        x=candles["time"],
        open=candles["open"], high=candles["high"],
        low=candles["low"], close=candles["close"],
        name="Price"
    ))
    # Volume Bars
    fig2.add_trace(go.Bar(
        x=candles["time"], y=candles["volume"],
        name="Volume", marker=dict(color="blue"), opacity=0.3, yaxis="y2"
    ))
    # Current Price Line
    fig2.add_hline(y=current_price, line_dash="dash", line_color="orange",
                   annotation_text="Current Price", annotation_position="top left")
    # Layout
    fig2.update_layout(
        title=f"BTC/USDT {interval} Candles",
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis2=dict(overlaying="y", side="right", showgrid=False, position=1),
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)

