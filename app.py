import time
import math
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every 1 second ---
st_autorefresh(interval=1000, key="refresh")  # 1000ms = 1s

# --- Page Setup ---
st.set_page_config(page_title="BTC/USDT Advanced Order Flow (Sniper)", layout="wide")
st.title("🔎 BTC/USDT Advanced Order Flow — Sniper Edition (Hidden Logic Enabled)")

# --- Session state init ---
if "wpi_history" not in st.session_state:
    st.session_state.wpi_history = []

if "prev_bids" not in st.session_state:
    st.session_state.prev_bids = None
if "prev_asks" not in st.session_state:
    st.session_state.prev_asks = None
if "fake_wall_events" not in st.session_state:
    st.session_state.fake_wall_events = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = {}

# --- User Inputs ---
with st.sidebar:
    st.header("Settings")
    use_binance_com = st.checkbox("Use binance.com (vs binance.us)", value=True)
    interval = st.selectbox("Candle Interval", ["1m", "5m", "15m", "1h", "4h"], index=0)
    candle_limit = st.number_input("Number of Candles", 10, 200, 50)
    orderbook_limit = st.selectbox("Orderbook Limit (rows)", [50, 100, 200], index=2)
    agressive_mode = st.checkbox("Aggressive sensitivity (higher false positives)", value=False)
    st.markdown("**Hidden strategy enabled:** Smart Money Imbalance, Delta WPI, Volume Spike, Liquidity Grab.")

# --- Helper: API base ---
API_BASE = "https://api.binance.com" if use_binance_com else "https://api.binance.us"

# --- Fetch Order Book with retries ---
def get_orderbook(limit=200, retries=2, timeout=3):
    url = API_BASE + "/api/v3/depth"
    params = {"symbol": "BTCUSDT", "limit": limit}
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
            asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)
            # Ensure sort order
            bids = bids.sort_values("price", ascending=False).reset_index(drop=True)
            asks = asks.sort_values("price", ascending=True).reset_index(drop=True)
            return bids, asks
        except Exception as e:
            if attempt == retries:
                return None, None
            time.sleep(0.2)
    return None, None

# --- Fetch candles with caching (short TTL via manual timestamp) ---
def get_candles(limit=50, interval="1m", retries=2, timeout=3):
    url = API_BASE + "/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data, columns=[
                "time","open","high","low","close","volume",
                "c","q","n","taker_base","taker_quote","ignore"
            ])
            # Convert numeric columns
            for col in ["open","high","low","close","volume","time"]:
                if col != "time":
                    df[col] = df[col].astype(float)
            df["time"] = pd.to_datetime(df["time"].astype(int), unit="ms")
            return df[["time","open","high","low","close","volume"]]
        except Exception as e:
            if attempt == retries:
                return None
            time.sleep(0.2)
    return None

# --- Get Data ---
bids, asks = get_orderbook(limit=orderbook_limit)
candles = get_candles(limit=candle_limit, interval=interval)

if bids is None or asks is None or candles is None:
    st.error("❌ Failed to fetch market data (API error). Try toggling binance.com / binance.us in sidebar.")
    st.stop()

# --- Current Market Price ---
current_price = float(candles["close"].iloc[-1])

# --- WPI calculation (weighted by inverse distance to current price) ---
# prevent zero distances by clipping to small epsilon
bids["price"] = bids["price"].astype(float)
asks["price"] = asks["price"].astype(float)
bids["qty"] = bids["qty"].astype(float)
asks["qty"] = asks["qty"].astype(float)

bid_dist = (current_price - bids["price"]).clip(lower=0.0001)
ask_dist = (asks["price"] - current_price).clip(lower=0.0001)

weighted_bids = (bids["qty"] / bid_dist).sum()
weighted_asks = (asks["qty"] / ask_dist).sum()

# avoid divide-by-zero
if weighted_bids + weighted_asks == 0:
    wpi = 0.0
else:
    wpi = (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks)

# store WPI history (time + value)
st.session_state.wpi_history.append({"time": pd.Timestamp.now(), "wpi": float(wpi)})
# keep limited history for trend calculations
st.session_state.wpi_history = st.session_state.wpi_history[-200:]
wpi_df = pd.DataFrame(st.session_state.wpi_history)

# --- Simple smoothed WPI trend (tail mean) ---
def wpi_trend(samples=3):
    if len(wpi_df) < samples:
        return 0.0
    return float(wpi_df["wpi"].tail(samples).mean())

# --- Volume Spike detection ---
def is_volume_spike(candles_df, multiplier=1.5):
    if len(candles_df) < 20:
        avg = candles_df["volume"].mean()
    else:
        avg = candles_df["volume"].tail(20).mean()
    current_vol = candles_df["volume"].iloc[-1]
    return current_vol > (avg * multiplier), current_vol, avg

vol_spike, current_vol, avg_vol = is_volume_spike(candles, multiplier=1.5 if not agressive_mode else 1.25)

# --- Fake wall (disappearing large order) detection ---
# Strategy: compare top N walls now vs previous snapshot, if a very large wall (relative to median) was present and now gone/ greatly reduced -> mark as possible fake
def detect_fake_walls(prev_df, curr_df, side="bids", threshold_factor=5.0):
    events = []
    if prev_df is None:
        return events
    # work on shallow copy
    prev = prev_df.copy()
    curr = curr_df.copy()
    # compute median qty to scale
    med = prev["qty"].median() if len(prev) > 0 else 0
    if med == 0:
        med = 1
    # find big walls in previous top 10
    prev_top = prev.head(10)
    for _, row in prev_top.iterrows():
        price = row["price"]
        qty = row["qty"]
        if qty > med * threshold_factor:
            # search curr for same price within small tolerance
            # exact price match is typical; allow small rounding issues
            match = curr[abs(curr["price"] - price) < 1e-8]
            if match.empty or float(match["qty"].iloc[0]) < qty * 0.2:
                events.append({"price": float(price), "qty": float(qty)})
    return events

fake_bids = detect_fake_walls(st.session_state.prev_bids, bids, side="bids", threshold_factor=4.5 if not agressive_mode else 3.0)
fake_asks = detect_fake_walls(st.session_state.prev_asks, asks, side="asks", threshold_factor=4.5 if not agressive_mode else 3.0)

# record fake wall events
for e in fake_bids:
    st.session_state.fake_wall_events.append({"side": "bid", **e, "time": pd.Timestamp.now()})
for e in fake_asks:
    st.session_state.fake_wall_events.append({"side": "ask", **e, "time": pd.Timestamp.now()})
# keep short history
st.session_state.fake_wall_events = st.session_state.fake_wall_events[-50:]

# --- Nearest bid/ask above/below price ---
nearest_bid = bids[bids["price"] < current_price].sort_values("price", ascending=False).head(1)
nearest_ask = asks[asks["price"] > current_price].sort_values("price", ascending=True).head(1)

if not nearest_bid.empty:
    nearest_bid_price, nearest_bid_qty = float(nearest_bid.iloc[0]["price"]), float(nearest_bid.iloc[0]["qty"])
else:
    nearest_bid_price, nearest_bid_qty = math.nan, 0.0

if not nearest_ask.empty:
    nearest_ask_price, nearest_ask_qty = float(nearest_ask.iloc[0]["price"]), float(nearest_ask.iloc[0]["qty"])
else:
    nearest_ask_price, nearest_ask_qty = math.nan, 0.0

# --- Liquidity grab / stop hunt detection ---
# Heuristic: last candle wick pierced a big nearby wall then closed opposite direction
def detect_liquidity_grab(candles_df, big_wall_price, side="bid", wick_ratio_threshold=0.6):
    # needs at least one previous candle
    if len(candles_df) < 1 or math.isnan(big_wall_price):
        return False
    last = candles_df.iloc[-1]
    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    # If checking bid (below price), price must have dipped to or below wall and closed green
    if side == "bid":
        dipped = l <= big_wall_price <= (o if o < c else c) + 1e-9  # somewhat below or equal
        # wick size downwards
        body = abs(c - o)
        lower_wick = (o - l) if o > l else (c - l)
        if body == 0:
            return False
        if dipped and c > o and (lower_wick / (body + lower_wick) > wick_ratio_threshold):
            return True
    else:  # ask side: price spiked above wall and closed red
        spiked = h >= big_wall_price >= (o if o > c else c) - 1e-9
        body = abs(c - o)
        upper_wick = (h - o) if o < h else (h - c)
        if body == 0:
            return False
        if spiked and c < o and (upper_wick / (body + upper_wick) > wick_ratio_threshold):
            return True
    return False

liquidity_grab_bid = detect_liquidity_grab(candles, nearest_bid_price, side="bid", wick_ratio_threshold=0.55 if not agressive_mode else 0.5)
liquidity_grab_ask = detect_liquidity_grab(candles, nearest_ask_price, side="ask", wick_ratio_threshold=0.55 if not agressive_mode else 0.5)

# --- Whale walls (largest qty) ---
big_bid = bids.loc[bids["qty"].idxmax()] if len(bids) > 0 else None
big_ask = asks.loc[asks["qty"].idxmax()] if len(asks) > 0 else None

# --- Strategy fusion: compute signals and confidence ---
score = 0.0
reasons = []

# WPI base
if wpi > 0.25:
    score += 30
    reasons.append("Bullish WPI")
elif wpi < -0.25:
    score -= 30
    reasons.append("Bearish WPI")

# WPI trend
trend_val = wpi_trend(samples=3)
if trend_val > 0.15:
    score += 15
    reasons.append("Rising WPI momentum")
elif trend_val < -0.15:
    score -= 15
    reasons.append("Falling WPI momentum")

# Volume spike
if vol_spike:
    # align direction: if last candle close>open and WPI>0 -> stronger, else partial
    last_candle = candles.iloc[-1]
    candle_dir = 1 if last_candle["close"] > last_candle["open"] else -1
    if candle_dir == 1 and wpi > 0:
        score += 20
        reasons.append("Volume spike confirming buy")
    elif candle_dir == -1 and wpi < 0:
        score -= 20
        reasons.append("Volume spike confirming sell")
    else:
        # volume spike but not aligned, small weight
        score += 5 * candle_dir
        reasons.append("Volume spike (mixed)")

# Fake wall detection: penalize side that had fake walls (they trap)
if len(fake_bids) > 0:
    # fake bid means likely trap -> bearish (sellers trick buyers by fake bid)
    score -= 18 * len(fake_bids)
    reasons.append(f"{len(fake_bids)} disappearing bid wall(s) - possible sell trap")
if len(fake_asks) > 0:
    score += 18 * len(fake_asks)
    reasons.append(f"{len(fake_asks)} disappearing ask wall(s) - possible buy trap")

# Liquidity grab adds strong signal opposite of the grab direction
if liquidity_grab_bid:
    # price dipped into bid and recovered -> bullish
    score += 25
    reasons.append("Liquidity grab at bid (stop hunt) — bullish")
if liquidity_grab_ask:
    score -= 25
    reasons.append("Liquidity grab at ask (stop hunt) — bearish")

# Nearest wall qty imbalance
wall_imbalance = 0.0
if nearest_bid_qty + nearest_ask_qty > 0:
    wall_imbalance = (nearest_bid_qty - nearest_ask_qty) / (nearest_bid_qty + nearest_ask_qty)
    # scale
    if wall_imbalance > 0.3:
        score += 8
        reasons.append("Larger nearest bid wall")
    elif wall_imbalance < -0.3:
        score -= 8
        reasons.append("Larger nearest ask wall")

# Big whale presence near price (extra caution; we prefer to follow it)
if big_bid is not None and abs(float(big_bid["price"]) - current_price) / current_price < 0.005:
    score += 6
    reasons.append("Large bid wall close by")
if big_ask is not None and abs(float(big_ask["price"]) - current_price) / current_price < 0.005:
    score -= 6
    reasons.append("Large ask wall close by")

# small tweak for aggressive setting
if agressive_mode:
    score *= 1.15

# Convert score to confidence and direction
# Score range roughly -100..+100; clamp
score = max(min(score, 100), -100)
abs_score = abs(score)
if abs_score < 10:
    prediction = "⚖️ Neutral / Range"
    confidence = int(abs_score)
elif score > 0:
    prediction = "🚀 Bullish Breakout / Long Bias"
    confidence = int(min(95, 30 + abs_score))  # base 30 + score
else:
    prediction = "📉 Bearish Breakdown / Short Bias"
    confidence = int(min(95, 30 + abs_score))

# Compose explanation (short)
explanation = "; ".join(reasons[:6]) if reasons else "No clear micro-structural signals."

# Persist last analysis
st.session_state.last_analysis = {
    "wpi": wpi,
    "score": score,
    "prediction": prediction,
    "confidence": confidence,
    "explanation": explanation,
    "vol_spike": vol_spike,
    "fake_bids": fake_bids,
    "fake_asks": fake_asks,
    "liquidity_grab_bid": liquidity_grab_bid,
    "liquidity_grab_ask": liquidity_grab_ask,
}

# --- Show metrics + Sniper Prediction ---
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric("Weighted Buy Pressure", f"{weighted_bids:,.2f}")
col3.metric("Weighted Sell Pressure", f"{weighted_asks:,.2f}")

# Sniper prediction box (prominent)
st.markdown("---")
pred_col, reason_col = st.columns([1, 2])
with pred_col:
    st.subheader("🎯 Sniper Prediction")
    st.markdown(f"### {prediction}")
    st.markdown(f"**Confidence:** {confidence}%")
    # Short reason
    st.write(explanation)
    # Quick actionable hint (hidden-style)
    if prediction.startswith("🚀"):
        st.success("Sniper hint: consider buys above current minor resistance; align with volume-confirmed candle.")
    elif prediction.startswith("📉"):
        st.error("Sniper hint: consider sells on break of nearest support; watch for fake-bid traps.")
    else:
        st.info("Sniper hint: range trade or wait for clear confirmation.")

with reason_col:
    st.subheader("🔍 Recent Strategy Signals (background)")
    st.write({
        "WPI": round(wpi, 4),
        "WPI_trend": round(trend_val, 4),
        "Volume Spike": vol_spike,
        "Liquidity Grab Bid": liquidity_grab_bid,
        "Liquidity Grab Ask": liquidity_grab_ask,
        "Fake Bids Detected": len(fake_bids),
        "Fake Asks Detected": len(fake_asks)
    })

st.caption("Note: Strategy heuristics are for informational/sniper-signal purposes. Use risk management. Do not blindly trade.")

# --- Layout with 3 Charts ---
col_left, col_center, col_right = st.columns([1, 1, 1])

# --- Depth Chart with whale markers ---
with col_left:
    bids["cum_qty"] = bids["qty"].cumsum()
    asks["cum_qty"] = asks["qty"].cumsum()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=bids["price"], y=bids["cum_qty"], mode="lines", name="Bid Depth"))
    fig1.add_trace(go.Scatter(x=asks["price"], y=asks["cum_qty"], mode="lines", name="Ask Depth"))
    # mark big walls
    if big_bid is not None:
        fig1.add_vline(x=float(big_bid["price"]), line_dash="dot", annotation_text=f"Big Bid {big_bid['qty']:.2f}", annotation_position="top left")
    if big_ask is not None:
        fig1.add_vline(x=float(big_ask["price"]), line_dash="dot", line_color="red", annotation_text=f"Big Ask {big_ask['qty']:.2f}", annotation_position="top right")
    # mark recent fake walls
    for ev in st.session_state.fake_wall_events[-6:]:
        color = "green" if ev["side"] == "bid" else "red"
        fig1.add_vline(x=ev["price"], line_dash="dash", line_color=color, annotation_text=f"Fake {ev['side']} {ev['qty']:.1f}", annotation_position="bottom right")
    fig1.update_layout(title="Order Book Cumulative Depth", xaxis_title="Price", yaxis_title="Cumulative Qty", height=480)
    st.plotly_chart(fig1, use_container_width=True)

# --- WPI Line Chart ---
with col_center:
    fig_wpi = go.Figure()
    bullish = wpi_df[wpi_df["wpi"] > 0.25]
    bearish = wpi_df[wpi_df["wpi"] < -0.25]
    neutral = wpi_df[(wpi_df["wpi"] >= -0.25) & (wpi_df["wpi"] <= 0.25)]
    if not neutral.empty:
        fig_wpi.add_trace(go.Scatter(x=neutral["time"], y=neutral["wpi"], mode="lines+markers", name="Neutral WPI"))
    if not bullish.empty:
        fig_wpi.add_trace(go.Scatter(x=bullish["time"], y=bullish["wpi"], mode="lines+markers", name="Bullish WPI"))
    if not bearish.empty:
        fig_wpi.add_trace(go.Scatter(x=bearish["time"], y=bearish["wpi"], mode="lines+markers", name="Bearish WPI"))
    fig_wpi.update_layout(title="Live Weighted Pressure Index (WPI)", xaxis_title="Time", yaxis_title="WPI", yaxis=dict(range=[-1,1]), height=480)
    st.plotly_chart(fig_wpi, use_container_width=True)

# --- Candlestick Chart with colored volume ---
with col_right:
    fig2 = go.Figure()
    # add candles
    fig2.add_trace(go.Candlestick(x=candles["time"], open=candles["open"], high=candles["high"], low=candles["low"], close=candles["close"], name="Price"))
    # compute color per candle
    colors = ["green" if o < c else "red" for o, c in zip(candles["open"], candles["close"])]
    fig2.add_trace(go.Bar(x=candles["time"], y=candles["volume"], name="Volume", marker_color=colors, opacity=0.35, yaxis="y2"))
    # current price line
    fig2.add_hline(y=current_price, line_dash="dash", annotation_text="Current Price", annotation_position="top left")
    # mark nearest walls
    if not math.isnan(nearest_bid_price):
        fig2.add_hline(y=nearest_bid_price, line_dash="dot", annotation_text="Nearest Bid", annotation_position="bottom left")
    if not math.isnan(nearest_ask_price):
        fig2.add_hline(y=nearest_ask_price, line_dash="dot", annotation_text="Nearest Ask", annotation_position="top right")
    # layout
    fig2.update_layout(title=f"BTC/USDT {interval} Candles", xaxis_title="Time", yaxis_title="Price", yaxis2=dict(overlaying="y", side="right", showgrid=False, position=1), height=480)
    st.plotly_chart(fig2, use_container_width=True)

# --- Optional: show recent fake wall events for inspection ---
if st.sidebar.checkbox("Show recent fake wall events", value=False):
    st.sidebar.write(pd.DataFrame(st.session_state.fake_wall_events)[["time","side","price","qty"]].tail(10))

# --- Update prev orderbook snapshot for next tick (store shallow top-rows only to save memory) ---
st.session_state.prev_bids = bids.head(40).copy()
st.session_state.prev_asks = asks.head(40).copy()

