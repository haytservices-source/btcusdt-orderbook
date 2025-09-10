import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every 1 second ---
st_autorefresh(interval=1000, key="refresh")

# --- Page Setup ---
st.set_page_config(page_title="BTC/USDT Advanced Order Flow", layout="wide")
st.title("📊 BTC/USDT Advanced Order Flow Dashboard")

# --- Initialize WPI history ---
if "wpi_history" not in st.session_state:
    st.session_state.wpi_history = []

# --- Sidebar ---
exchange = st.sidebar.radio("Select Exchange", ["binance.com", "binance.us"], index=0)
base_url = "https://api." + exchange

interval = st.sidebar.selectbox("Select Candle Interval", ["1m", "5m", "15m", "1h", "4h"], index=0)
candle_limit = st.sidebar.number_input("Number of Candles", 20, 200, 50)

# --- Fetch Order Book ---
def get_orderbook(limit=200):
    url = f"{base_url}/api/v3/depth"
    params = {"symbol": "BTCUSDT", "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        data = res.json()
        bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)
        return bids, asks
    except Exception as e:
        st.error(f"❌ Order book fetch error: {e}")
        return None, None

# --- Fetch Candlestick Data ---
def get_candles(limit=50, interval="1m"):
    url = f"{base_url}/api/v3/klines"
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
    except Exception as e:
        st.error(f"❌ Candles fetch error: {e}")
        return None

# --- Get Data ---
bids, asks = get_orderbook()
candles = get_candles(limit=candle_limit, interval=interval)

if bids is None or asks is None or candles is None:
    st.error("❌ Failed to fetch market data (API error). Try toggling binance.com / binance.us in sidebar.")
    st.stop()

# --- Current Market Price ---
current_price = candles["close"].iloc[-1]

# --- Weighted Pressure Calculation ---
bid_dist = (current_price - bids["price"]).clip(lower=0.01)
ask_dist = (asks["price"] - current_price).clip(lower=0.01)

weighted_bids = (bids["qty"] / bid_dist).sum()
weighted_asks = (asks["qty"] / ask_dist).sum()

# --- Weighted Pressure Index (WPI) ---
wpi = (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks)
st.session_state.wpi_history.append({"time": pd.Timestamp.now(), "wpi": wpi})
st.session_state.wpi_history = st.session_state.wpi_history[-100:]
wpi_df = pd.DataFrame(st.session_state.wpi_history)

# --- Market Bias ---
if wpi > 0.25:
    bias = "🔥 Buyers Dominant (Bullish)"
elif wpi < -0.25:
    bias = "🔴 Sellers Dominant (Bearish)"
else:
    bias = "⚖️ Neutral / Sideways"

# --- Whale Detection ---
big_bid = bids.loc[bids["qty"].idxmax()]
big_ask = asks.loc[asks["qty"].idxmax()]

nearest_bid = bids[bids["price"] < current_price].sort_values("price", ascending=False).head(1)
nearest_ask = asks[asks["price"] > current_price].sort_values("price", ascending=True).head(1)

nearest_bid_price, nearest_bid_qty = nearest_bid.iloc[0]["price"], nearest_bid.iloc[0]["qty"]
nearest_ask_price, nearest_ask_qty = nearest_ask.iloc[0]["price"], nearest_ask.iloc[0]["qty"]

# --- Secret Strategy Logic ---
prediction = "🤔 Unclear"
confidence = 50
reason = "No strong signal detected."

# Delta WPI (momentum)
if len(wpi_df) > 5:
    delta_wpi = wpi_df["wpi"].iloc[-1] - wpi_df["wpi"].iloc[-5]
else:
    delta_wpi = 0

# Volume spike check
avg_vol = candles["volume"].iloc[-20:].mean()
vol_spike = candles["volume"].iloc[-1] > 1.5 * avg_vol

# Stop hunt detection
last_candle = candles.iloc[-1]
wick_down = last_candle["low"] < nearest_bid_price and last_candle["close"] > last_candle["open"]
wick_up = last_candle["high"] > nearest_ask_price and last_candle["close"] < last_candle["open"]

if wpi > 0.2 and delta_wpi > 0.05 and vol_spike:
    prediction = "🚀 Bullish Breakout Incoming"
    confidence = min(95, abs(wpi) * 100)
    reason = "WPI rising + strong buy pressure + volume spike."
elif wpi < -0.2 and delta_wpi < -0.05 and vol_spike:
    prediction = "📉 Bearish Breakdown Likely"
    confidence = min(95, abs(wpi) * 100)
    reason = "WPI falling + strong sell pressure + volume spike."
elif wick_down:
    prediction = "🟢 Bullish Liquidity Grab (Stop Hunt)"
    confidence = 75
    reason = "Price pierced bid wall but closed green."
elif wick_up:
    prediction = "🔴 Bearish Liquidity Grab (Stop Hunt)"
    confidence = 75
    reason = "Price pierced ask wall but closed red."

# --- Show Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric("Weighted Buy Pressure", f"{weighted_bids:,.2f}")
col3.metric("Weighted Sell Pressure", f"{weighted_asks:,.2f}")

# --- Prediction Box ---
st.subheader(f"🎯 Sniper Prediction → {prediction}")
st.write(f"**Reason:** {reason}")
st.write(f"**Confidence:** {confidence:.1f}%")
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

# --- WPI Line Chart ---
with col_center:
    fig_wpi = go.Figure()
    fig_wpi.add_trace(go.Scatter(
        x=wpi_df["time"], y=wpi_df["wpi"], mode="lines+markers",
        name="WPI", line=dict(color="blue")
    ))
    fig_wpi.update_layout(
        title="Live Weighted Pressure Index (WPI)",
        xaxis_title="Time", yaxis_title="WPI",
        yaxis=dict(range=[-1,1]),
        height=500
    )
    st.plotly_chart(fig_wpi, use_container_width=True)

# --- Candlestick Chart with Volume ---
with col_right:
    fig2 = go.Figure()
    fig2.add_trace(go.Candlestick(
        x=candles["time"],
        open=candles["open"], high=candles["high"],
        low=candles["low"], close=candles["close"],
        name="Price"
    ))
    fig2.add_trace(go.Bar(
        x=candles["time"], y=candles["volume"],
        name="Volume", marker=dict(color="blue"), opacity=0.3, yaxis="y2"
    ))
    fig2.add_hline(y=current_price, line_dash="dash", line_color="orange",
                   annotation_text="Current Price", annotation_position="top left")
    fig2.update_layout(
        title=f"BTC/USDT {interval} Candles",
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis2=dict(overlaying="y", side="right", showgrid=False, position=1),
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)
