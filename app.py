import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="BTC/USDT Pro Order Flow", layout="wide")
st_autorefresh(interval=2000, key="refresh")  # refresh every 2 seconds
st.title("📊 BTC/USDT Pro Order Flow Dashboard")

# --------------------------------
# SIDEBAR CONTROLS
# --------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    exchange = st.selectbox("Data Source (Spot)", ["binance.us", "binance.com"], index=0)
    depth_limit = st.selectbox("Order Book Depth", [50, 100, 200, 500, 1000], index=2)
    top_n = st.slider("Top N Levels for Imbalance", 10, 100, 30, step=5)
    heatmap_bins = st.slider("Heatmap Price Bin Count", 30, 150, 60, step=10)
    w_near_decay = st.slider("Weighting Decay (near-mid emphasis)", 0.5, 4.0, 1.6, step=0.1)
    show_trades = st.checkbox("Show Recent Trades (tape)", value=True)
    st.caption("Tip: Higher decay emphasizes levels closer to mid price.")

# -------------------------------
# HELPERS
# -------------------------------
def base_url():
    # Spot market endpoints (public)
    return "https://api.binance.us" if exchange == "binance.us" else "https://api.binance.com"

def get_orderbook(symbol="BTCUSDT", limit=200):
    url = f"{base_url()}/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        data = res.json()
        bids = pd.DataFrame(data.get("bids", []), columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(data.get("asks", []), columns=["price", "qty"], dtype=float)
        return bids, asks
    except Exception:
        return None, None

def get_klines(interval="1m", limit=100):
    url = f"{base_url()}/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        data = res.json()
        cols = ["time","open","high","low","close","volume","c","q","n","taker_base","taker_quote","ignore"]
        df = pd.DataFrame(data, columns=cols, dtype=float)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df[["time","open","high","low","close","volume"]]
    except Exception:
        return None

def get_trades(limit=200):
    url = f"{base_url()}/api/v3/trades"
    params = {"symbol": "BTCUSDT", "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        raw = res.json()
        # Normalize
        df = pd.DataFrame(raw)
        if df.empty:
            return pd.DataFrame()
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["price"] = pd.to_numeric(df["price"])
        df["qty"] = pd.to_numeric(df["qty"])
        # 'isBuyerMaker' True means seller initiated (hit the bid) -> sell
        df["side"] = np.where(df["isBuyerMaker"], "SELL", "BUY")
        return df[["time","price","qty","side"]].sort_values("time")
    except Exception:
        return pd.DataFrame()

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def weighted_pressure(bids, asks, mid, decay=1.6):
    """
    Heavier weight for levels closer to mid price using exponential decay on distance.
    weight = qty / (1 + |price-mid|^decay)
    """
    if bids.empty or asks.empty:
        return 0.0, 0.0, 0.0

    # Avoid division by zero and negative power quirks
    b = bids.copy()
    a = asks.copy()
    b["dist"] = (mid - b["price"]).clip(lower=1e-6)
    a["dist"] = (a["price"] - mid).clip(lower=1e-6)

    b["w"] = b["qty"] / (1.0 + np.power(b["dist"], decay))
    a["w"] = a["qty"] / (1.0 + np.power(a["dist"], decay))

    wb = b["w"].sum()
    wa = a["w"].sum()
    denom = wb + wa
    wpi = (wb - wa) / denom if denom > 0 else 0.0
    return wb, wa, wpi

def top_n_imbalance(bids, asks, mid, n):
    # Focus on nearest N levels around mid (N/2 each side if available)
    nb = min(len(bids), n)
    na = min(len(asks), n)
    b_sum = bids.sort_values("price", ascending=False).head(nb)["qty"].sum()
    a_sum = asks.sort_values("price", ascending=True).head(na)["qty"].sum()
    denom = b_sum + a_sum
    return (b_sum - a_sum) / denom if denom > 0 else 0.0, b_sum, a_sum

def nearest_levels(bids, asks, mid):
    nb = bids[bids["price"] < mid].sort_values("price", ascending=False).head(1)
    na = asks[asks["price"] > mid].sort_values("price", ascending=True).head(1)
    if nb.empty or na.empty:
        return None
    return dict(
        bid_price=float(nb.iloc[0]["price"]),
        bid_qty=float(nb.iloc[0]["qty"]),
        ask_price=float(na.iloc[0]["price"]),
        ask_qty=float(na.iloc[0]["qty"])
    )

def liquidity_shift(new_bids, new_asks, old_bids, old_asks, band_bps=10):
    """
    Compare current vs previous snapshot near mid (±band_bps basis points).
    Positive means added bid liquidity; negative means removed (potential spoof).
    """
    if new_bids is None or new_asks is None or old_bids is None or old_asks is None:
        return 0.0, 0.0

    mid_new = (new_bids["price"].max() + new_asks["price"].min()) / 2.0
    band = mid_new * band_bps / 10000.0  # basis points band

    def sum_band(df, side):
        if side == "bid":
            mask = (df["price"] >= mid_new - band) & (df["price"] < mid_new)
        else:
            mask = (df["price"] <= mid_new + band) & (df["price"] > mid_new)
        return df.loc[mask, "qty"].sum()

    # Old mid for fair band? We'll use new mid for both to simplify.
    new_bid_q = sum_band(new_bids, "bid")
    new_ask_q = sum_band(new_asks, "ask")
    old_bid_q = sum_band(old_bids, "bid")
    old_ask_q = sum_band(old_asks, "ask")

    return (new_bid_q - old_bid_q), (new_ask_q - old_ask_q)

# --------------------------------
# FETCH DATA
# --------------------------------
bids, asks = get_orderbook(limit=depth_limit)
m1 = get_klines("1m", limit=120)   # 2 hours
m15 = get_klines("15m", limit=200) # trend context

if bids is None or asks is None or m1 is None or m15 is None or bids.empty or asks.empty:
    st.error("❌ Failed to fetch market data.")
    st.stop()

mid_price = (bids["price"].max() + asks["price"].min()) / 2.0

# Keep previous snapshot for liquidity shift
if "prev_bids" not in st.session_state:
    st.session_state.prev_bids = bids.copy()
    st.session_state.prev_asks = asks.copy()

# --------------------------------
# CORE METRICS
# --------------------------------
wb, wa, wpi = weighted_pressure(bids, asks, mid_price, decay=w_near_decay)
imbalance, top_b_sum, top_a_sum = top_n_imbalance(bids, asks, mid_price, top_n)
near = nearest_levels(bids, asks, mid_price) or {}
big_bid = bids.loc[bids["qty"].idxmax()]
big_ask = asks.loc[asks["qty"].idxmax()]
d_bid, d_ask = liquidity_shift(bids, asks, st.session_state.prev_bids, st.session_state.prev_asks)

# Tape & absorption
tape = get_trades(300) if show_trades else pd.DataFrame()
absorption_note = "n/a"
if not tape.empty and len(m1) >= 5:
    # classify recent 30s
    recent = tape[tape["time"] >= (tape["time"].max() - pd.Timedelta(seconds=30))]
    buy_vol = recent.loc[recent["side"]=="BUY","qty"].sum()
    sell_vol = recent.loc[recent["side"]=="SELL","qty"].sum()

    # price move last 30s
    px_30s_start = m1.iloc[-1]["close"]
    # estimate last trade price movement vs last candle
    last_px = float(tape["price"].iloc[-1]) if not tape.empty else m1.iloc[-1]["close"]
    delta_px = last_px - px_30s_start

    # Simple absorption heuristic
    if buy_vol > sell_vol * 1.5 and delta_px <= 0:
        absorption_note = "🧱 Seller absorption (buyers aggressive, price capped)"
    elif sell_vol > buy_vol * 1.5 and delta_px >= 0:
        absorption_note = "🧱 Buyer absorption (sellers aggressive, price held)"
    else:
        absorption_note = "Balanced / unclear"

# Multi-TF bias
m1["ema_fast"] = ema(m1["close"], 12)
m1["ema_slow"] = ema(m1["close"], 26)
m15["ema_fast"] = ema(m15["close"], 12)
m15["ema_slow"] = ema(m15["close"], 26)

m1_trend = "UP" if m1["ema_fast"].iloc[-1] > m1["ema_slow"].iloc[-1] else "DOWN"
m15_trend = "UP" if m15["ema_fast"].iloc[-1] > m15["ema_slow"].iloc[-1] else "DOWN"

# Bias from WPI
if wpi > 0.25:
    bias = "🔥 Buyers Dominant (Bullish)"
elif wpi < -0.25:
    bias = "🔴 Sellers Dominant (Bearish)"
else:
    bias = "⚖️ Neutral / Sideways"

# Simple signal combiner
score = 0.0
score += np.clip(wpi, -1, 1) * 2.0           # order flow weight
score += np.clip(imbalance, -1, 1) * 1.2     # book imbalance
score += 0.5 if d_bid > 0 else (-0.5 if d_bid < 0 else 0)  # bid liquidity shift
score += -0.5 if d_ask > 0 else (0.5 if d_ask < 0 else 0)  # ask liquidity shift
score += 0.6 if m1_trend == "UP" else -0.6
score += 0.8 if m15_trend == "UP" else -0.8

signal = "LONG" if score > 0.8 else ("SHORT" if score < -0.8 else "NEUTRAL")
confidence = f"{min(100, int(abs(score) / 3.5 * 100))}%"

# Projection
if near:
    if near["bid_qty"] > near["ask_qty"] and wpi > 0:
        projection = f"📈 Likely Up → Next Ask Zone ~ ${near['ask_price']:,.0f}"
    elif near["ask_qty"] > near["bid_qty"] and wpi < 0:
        projection = f"📉 Likely Down → Next Bid Zone ~ ${near['bid_price']:,.0f}"
    else:
        projection = f"🤔 Range ${near['bid_price']:,.0f} - ${near['ask_price']:,.0f}"
else:
    projection = "🤔 Projection unavailable (missing nearest levels)"

# -------------------------------
# TOP METRICS
# -------------------------------
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Mid Price", f"${mid_price:,.2f}", help=f"Source: {exchange}")
mcol2.metric("Weighted Buy Pressure", f"{wb:,.2f}")
mcol3.metric("Weighted Sell Pressure", f"{wa:,.2f}")
mcol4.metric("Order Book Imbalance (Top N)", f"{imbalance:+.2f}", help=f"N={top_n}")

st.subheader(f"Bias → {bias} | Multi-TF: 1m {m1_trend} / 15m {m15_trend}")
st.subheader(f"Signal → **{signal}** | Confidence: {confidence}")
st.caption(
    f"🐋 Biggest Buy Wall: {big_bid['qty']:.2f} BTC @ ${big_bid['price']:.0f} | "
    f"🐋 Biggest Sell Wall: {big_ask['qty']:.2f} BTC @ ${big_ask['price']:.0f} | "
    f"ΔBid Liquidity (near mid): {d_bid:+.2f} | ΔAsk Liquidity (near mid): {d_ask:+.2f}"
)
st.caption(f"Absorption: {absorption_note}")

st.markdown(f"**Projection** → {projection}")

# -------------------------------
# CHARTS
# -------------------------------
left, right = st.columns(2)

# Heatmap-like bar (binned) for depth
with left:
    # Bin prices to make a smoother heatmap bar view
    all_levels = pd.concat([
        bids.assign(side="BID"),
        asks.assign(side="ASK")
    ], ignore_index=True)

    # Create bins around current visible range
    pmin = min(bids["price"].min(), asks["price"].min())
    pmax = max(bids["price"].max(), asks["price"].max())
    bins = np.linspace(pmin, pmax, heatmap_bins)
    all_levels["bin"] = pd.cut(all_levels["price"], bins, include_lowest=True)
    depth_by_bin = all_levels.groupby(["bin","side"], observed=True)["qty"].sum().reset_index()
    # For x labels use bin midpoints
    bin_centers = [interval.mid for interval in depth_by_bin["bin"].cat.categories]
    depth_by_bin["bin_mid"] = depth_by_bin["bin"].apply(lambda iv: iv.mid)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=depth_by_bin.loc[depth_by_bin["side"]=="BID","bin_mid"],
        y=depth_by_bin.loc[depth_by_bin["side"]=="BID","qty"],
        name="Bids",
        marker=dict(color="green"),
        opacity=0.6
    ))
    fig1.add_trace(go.Bar(
        x=depth_by_bin.loc[depth_by_bin["side"]=="ASK","bin_mid"],
        y=depth_by_bin.loc[depth_by_bin["side"]=="ASK","qty"],
        name="Asks",
        marker=dict(color="red"),
        opacity=0.6
    ))
    fig1.update_layout(
        title="Liquidity Heatmap (Binned Order Book Depth)",
        xaxis_title="Price",
        yaxis_title="Quantity",
        barmode="overlay",
        height=520
    )
    st.plotly_chart(fig1, use_container_width=True, key="heatmap_chart")

# Candles with EMAs
with right:
    fig2 = go.Figure()
    fig2.add_trace(go.Candlestick(
        x=m1["time"], open=m1["open"], high=m1["high"], low=m1["low"], close=m1["close"], name="Price"
    ))
    fig2.add_trace(go.Scatter(x=m1["time"], y=m1["ema_fast"], name="EMA 12"))
    fig2.add_trace(go.Scatter(x=m1["time"], y=m1["ema_slow"], name="EMA 26"))
    fig2.update_layout(
        title="BTC/USDT — 1m Candles + EMAs",
        xaxis_title="Time",
        yaxis_title="Price",
        height=520
    )
    st.plotly_chart(fig2, use_container_width=True, key="candles_chart")

# Optional: Recent Trades (tape)
if show_trades:
    st.subheader("🧾 Recent Trades (last ~300 prints)")
    if tape.empty:
        st.info("No trades fetched.")
    else:
        # Small summary
        last_secs = tape[tape["time"] >= (tape["time"].max() - pd.Timedelta(seconds=30))]
        buy_v = last_secs[last_secs["side"]=="BUY"]["qty"].sum()
        sell_v = last_secs[last_secs["side"]=="SELL"]["qty"].sum()
        tcol1, tcol2, tcol3 = st.columns(3)
        tcol1.metric("30s BUY Volume", f"{buy_v:.4f}")
        tcol2.metric("30s SELL Volume", f"{sell_v:.4f}")
        tcol3.metric("Net (BUY-SELL)", f"{(buy_v - sell_v):+.4f}")

        fig3 = go.Figure()
        buys = tape[tape["side"]=="BUY"]
        sells = tape[tape["side"]=="SELL"]
        fig3.add_trace(go.Scatter(
            x=buys["time"], y=buys["price"], mode="markers",
            marker=dict(size=np.clip(buys["qty"]*2, 4, 18), color="green"),
            name="Aggressive BUY"
        ))
        fig3.add_trace(go.Scatter(
            x=sells["time"], y=sells["price"], mode="markers",
            marker=dict(size=np.clip(sells["qty"]*2, 4, 18), color="red"),
            name="Aggressive SELL"
        ))
        fig3.update_layout(
            title="Time & Sales (bubble size ~ trade size)",
            xaxis_title="Time",
            yaxis_title="Price",
            height=420
        )
        st.plotly_chart(fig3, use_container_width=True, key="tape_chart")

# --------------------------------
# FOOTER / SAFETY
# --------------------------------
ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.caption(f"Last update: {ts} | Source: {exchange} spot public APIs | Not financial advice.")

# Update previous snapshot for next refresh
st.session_state.prev_bids = bids.copy()
st.session_state.prev_asks = asks.copy()
