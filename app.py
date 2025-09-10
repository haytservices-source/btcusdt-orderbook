# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
import time
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- Auto-refresh every 2 seconds ---
st_autorefresh(interval=2000, key="refresh")

st.set_page_config(page_title="BTC/USDT Smart Order Flow (Pro)", layout="wide")
st.title("🔐 BTC/USDT Smart Order Flow — Pro (EMA / RSI / Whale / Liquidity)")

# ------------------------
# -------- SETTINGS -------
# ------------------------
BASE_URL = "https://api.binance.us"
SYMBOL = "BTCUSDT"
ORDERBOOK_LIMIT = 200
# Lookback candles for indicators
LOOKBACK_1M = 200
LOOKBACK_HIGHER = 200

# Sidebar options
with st.sidebar:
    st.header("Settings")
    aggressive = st.checkbox("Aggressive Mode (more sensitive)", value=False)
    show_debug = st.checkbox("Show debug details (orderbook, components)", value=False)
    ema_short_len = st.number_input("EMA short (fast) length", min_value=10, max_value=200, value=50)
    ema_long_len = st.number_input("EMA long (slow) length", min_value=50, max_value=400, value=200)
    rsi_len = st.number_input("RSI length", min_value=7, max_value=30, value=14)
    st.caption("App auto-reads 1m / 5m / 15m / 1h and fuses signals.")

# ------------------------
# ---- Helper funcs ------
# ------------------------
def fetch_with_retries(url, params=None, retries=2, timeout=4):
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.15)
    raise last_err

def get_orderbook(limit=ORDERBOOK_LIMIT):
    url = f"{BASE_URL}/api/v3/depth"
    params = {"symbol": SYMBOL, "limit": limit}
    data = fetch_with_retries(url, params=params, retries=2)
    bids = pd.DataFrame(data["bids"], columns=["price","qty"], dtype=float)
    asks = pd.DataFrame(data["asks"], columns=["price","qty"], dtype=float)
    bids = bids.sort_values("price", ascending=False).reset_index(drop=True)
    asks = asks.sort_values("price", ascending=True).reset_index(drop=True)
    return bids, asks

def get_candles(interval="1m", limit=LOOKBACK_1M):
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": interval, "limit": limit}
    data = fetch_with_retries(url, params=params, retries=2)
    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "c","q","n","taker_base","taker_quote","ignore"
    ])
    # convert types
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df[["time","open","high","low","close","volume"]]

# EMA using pandas ewm
def ema(series, length):
    if len(series) < 2:
        return np.nan
    return series.ewm(span=length, adjust=False).mean()

# RSI (classic)
def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series

# Volume spike detection (uses last N candles)
def is_volume_spike(df, lookback=20, multiplier=1.5):
    if len(df) < lookback + 1:
        return False, None, None
    avg = df["volume"].iloc[-(lookback+1):-1].mean()
    recent = df["volume"].iloc[-1]
    return recent > avg * multiplier, recent, avg

# Liquidity grab detection
def liquidity_grab_last_candle(df, wall_price, side="bid", wick_ratio_threshold=0.6):
    # side == 'bid' means price dipped to wall and recovered (bullish)
    if len(df) < 1 or pd.isna(wall_price):
        return False
    last = df.iloc[-1]
    o,h,l,c = last["open"], last["high"], last["low"], last["close"]
    body = abs(c - o)
    if body == 0:
        return False
    if side == "bid":
        dipped = l <= wall_price
        lower_wick = (o - l) if o > l else (c - l)
        return dipped and (c > o) and (lower_wick / (body + lower_wick) > wick_ratio_threshold)
    else:
        spiked = h >= wall_price
        upper_wick = (h - o) if o < h else (h - c)
        return spiked and (c < o) and (upper_wick / (body + upper_wick) > wick_ratio_threshold)

# compute orderbook pressures
def orderbook_pressure(bids, asks, current_price):
    # weighted by inverse distance to current price (short-term pressure)
    bid_dist = (current_price - bids["price"]).clip(lower=1e-6)
    ask_dist = (asks["price"] - current_price).clip(lower=1e-6)
    weighted_bids = (bids["qty"] / bid_dist).sum()
    weighted_asks = (asks["qty"] / ask_dist).sum()
    total = weighted_bids + weighted_asks
    imbalance = 0.0
    if total != 0:
        imbalance = (weighted_bids - weighted_asks) / total  # -1..1
    # also compute simple price*qty pressure
    buy_pressure = (bids["price"] * bids["qty"]).sum()
    sell_pressure = (asks["price"] * asks["qty"]).sum()
    return weighted_bids, weighted_asks, imbalance, buy_pressure, sell_pressure

# whale wall detection: largest qty on each side and proximity
def whale_walls(bids, asks, current_price, proximity_pct=0.005):
    big_bid = bids.loc[bids["qty"].idxmax()] if len(bids)>0 else None
    big_ask = asks.loc[asks["qty"].idxmax()] if len(asks)>0 else None
    near_bid = False
    near_ask = False
    if big_bid is not None:
        near_bid = abs(big_bid["price"] - current_price) / current_price <= proximity_pct
    if big_ask is not None:
        near_ask = abs(big_ask["price"] - current_price) / current_price <= proximity_pct
    return big_bid, big_ask, near_bid, near_ask

# combine signals into confidence
def compute_confidence(components):
    """
    components: dict with boolean / numeric elements and weights
    We'll compute a weighted score 0..100
    """
    score = 0.0
    max_score = 0.0
    # weights (tunable)
    weights = {
        "orderbook_imbalance": 30,
        "trend_alignment": 20,
        "ema_alignment": 12,
        "rsi_alignment": 8,
        "volume_confirm": 15,
        "liquidity_grab": 10,
        "whale_near": 5
    }
    # orderbook_imbalance: -1..1 -> map to 0..weights
    imb = components.get("imbalance", 0.0)
    max_score += weights["orderbook_imbalance"]
    score += (abs(imb) * weights["orderbook_imbalance"])
    # trend_alignment: integer -1/0/1 (count of aligned timeframes normalized)
    max_score += weights["trend_alignment"]
    trend_frac = components.get("trend_frac", 0.0)  # 0..1
    score += trend_frac * weights["trend_alignment"]
    # ema_alignment: 0..1
    max_score += weights["ema_alignment"]
    score += (1.0 if components.get("ema_ok", False) else 0.0) * weights["ema_alignment"]
    # rsi alignment: 0..1 (1 if supports direction)
    max_score += weights["rsi_alignment"]
    score += (1.0 if components.get("rsi_ok", False) else 0.0) * weights["rsi_alignment"]
    # volume confirm: 0..1
    max_score += weights["volume_confirm"]
    score += (1.0 if components.get("volume_ok", False) else 0.0) * weights["volume_confirm"]
    # liquidity grab: boost (if positive)
    max_score += weights["liquidity_grab"]
    score += (1.0 if components.get("liquidity_boost", False) else 0.0) * weights["liquidity_grab"]
    # whale near: small boost
    max_score += weights["whale_near"]
    score += (1.0 if components.get("whale_near", False) else 0.0) * weights["whale_near"]

    # penalize if volume divergence (volume supports opposite)
    if components.get("volume_divergence", False):
        score *= 0.5

    # map to 0..100
    if max_score == 0:
        return 0
    pct = max(0.0, min(100.0, (score / max_score) * 100.0))
    # aggressive mode tweak
    if aggressive:
        pct = min(100.0, pct * 1.1)
    return int(pct)

# ------------------------
# ---- Main execution ----
# ------------------------
try:
    # Fetch multi-timeframe candles
    candles_1m = get_candles("1m", limit=LOOKBACK_1M)
    candles_5m = get_candles("5m", limit=LOOKBACK_HIGHER)
    candles_15m = get_candles("15m", limit=LOOKBACK_HIGHER)
    candles_1h = get_candles("1h", limit=LOOKBACK_HIGHER)
except Exception as e:
    st.error(f"Failed to fetch candles: {e}")
    st.stop()

# Orderbook
try:
    bids, asks = get_orderbook(ORDERBOOK_LIMIT)
except Exception as e:
    st.error(f"Failed to fetch orderbook: {e}")
    st.stop()

# Basic current price
current_price = float(candles_1m["close"].iloc[-1])

# Indicators: EMA & RSI (1m primary)
ema_short_1m = ema(candles_1m["close"], ema_short_len)
ema_long_1m = ema(candles_1m["close"], ema_long_len)
ema_short_1h = ema(candles_1h["close"], ema_short_len) if candles_1h is not None else None
ema_long_1h = ema(candles_1h["close"], ema_long_len) if candles_1h is not None else None

rsi_1m = rsi(candles_1m["close"], rsi_len)
rsi_5m = rsi(candles_5m["close"], rsi_len)
rsi_15m = rsi(candles_15m["close"], rsi_len)
rsi_1h = rsi(candles_1h["close"], rsi_len)

# trend per timeframe (last N closes)
def tf_trend(candles_df, lookback=5):
    if candles_df is None or len(candles_df) < lookback:
        return 0
    return 1 if candles_df["close"].iloc[-1] > candles_df["close"].iloc[-lookback] else -1

trend_1m = tf_trend(candles_1m, lookback=5)
trend_5m = tf_trend(candles_5m, lookback=5)
trend_15m = tf_trend(candles_15m, lookback=5)
trend_1h = tf_trend(candles_1h, lookback=5)

trends = [trend_1m, trend_5m, trend_15m, trend_1h]
bullish_count = trends.count(1)
bearish_count = trends.count(-1)
trend_frac = abs(bullish_count - bearish_count) / len(trends)  # 0..1 measure of agreement

# Orderbook pressure & imbalance
weighted_bids, weighted_asks, imbalance, buy_pressure, sell_pressure = orderbook_pressure(bids, asks, current_price)

# Whale walls
big_bid, big_ask, near_big_bid, near_big_ask = whale_walls(bids, asks, current_price, proximity_pct=0.005)

# Volume spike primary (1m) and volume divergence
vol_spike_1m, vol_recent, vol_avg = is_volume_spike(candles_1m, lookback=20, multiplier=1.5 if not aggressive else 1.25)
# volume direction: last candle direction
last = candles_1m.iloc[-1]
last_dir = 1 if last["close"] > last["open"] else -1 if last["close"] < last["open"] else 0

# Check if volume supports orderbook imbalance direction:
imbalance_dir = 1 if imbalance > 0.05 else -1 if imbalance < -0.05 else 0
volume_divergence = False
if vol_spike_1m and imbalance_dir != 0 and last_dir != imbalance_dir:
    volume_divergence = True

# Liquidity grab detection
liq_grab_bid = liquidity_grab_last_candle(candles_1m, big_bid["price"] if big_bid is not None else np.nan, side="bid", wick_ratio_threshold=0.55)
liq_grab_ask = liquidity_grab_last_candle(candles_1m, big_ask["price"] if big_ask is not None else np.nan, side="ask", wick_ratio_threshold=0.55)
liquidity_boost = liq_grab_bid or liq_grab_ask

# EMA alignment: 1m close relative to EMA50/200 and 1h trend relative to EMA
ema_ok = False
if not math.isnan(ema_short_1m.iloc[-1]) and not math.isnan(ema_long_1m.iloc[-1]):
    if current_price > ema_short_1m.iloc[-1] and current_price > ema_long_1m.iloc[-1]:
        ema_ok = True  # bullish alignment
    elif current_price < ema_short_1m.iloc[-1] and current_price < ema_long_1m.iloc[-1]:
        ema_ok = True  # bearish alignment (alignment supports a short)
    else:
        ema_ok = False

# RSI check: support for direction (not overbought/oversold against signal)
# We'll consider RSI ok if it's not extreme opposite of imbalance/trend.
rsi_ok = False
rsi_val_1m = rsi_1m.iloc[-1] if len(rsi_1m)>0 else 50
if imbalance_dir == 1 and rsi_val_1m < 75:  # buyers and not super overbought
    rsi_ok = True
elif imbalance_dir == -1 and rsi_val_1m > 25:
    rsi_ok = True
else:
    # if no strong imbalance, check general RSI neutrality
    rsi_ok = 30 < rsi_val_1m < 70

# Components assemble
components = {
    "imbalance": imbalance,                 # -1..1
    "trend_frac": trend_frac,               # 0..1
    "ema_ok": ema_ok,                       # bool
    "rsi_ok": rsi_ok,                       # bool
    "volume_ok": (vol_spike_1m and not volume_divergence), # bool
    "liquidity_boost": liquidity_boost,     # bool
    "whale_near": (near_big_bid or near_big_ask),
    "volume_divergence": volume_divergence
}

confidence = compute_confidence({
    "imbalance": components["imbalance"],
    "trend_frac": components["trend_frac"],
    "ema_ok": components["ema_ok"],
    "rsi_ok": components["rsi_ok"],
    "volume_ok": components["volume_ok"],
    "liquidity_boost": components["liquidity_boost"],
    "whale_near": components["whale_near"],
    "volume_divergence": components["volume_divergence"]
})

# Final signal decision
# rules:
# - Strong BUY if majority trends bullish, imbalance positive, ema_ok true, volume confirms (and not volume_divergence)
# - Strong SELL if majority trends bearish, imbalance negative, ema_ok true, volume confirms (and not volume_divergence)
# - otherwise sideways or weak
signal = "SIDEWAYS ➡️"
if bullish_count > bearish_count and imbalance > 0.03 and components["ema_ok"] and components["volume_ok"]:
    signal = "BUY ⬆️"
elif bearish_count > bullish_count and imbalance < -0.03 and components["ema_ok"] and components["volume_ok"]:
    signal = "SELL ⬇️"
# liquidity grab can flip/strengthen regardless of trend counts
if liq_grab_bid and confidence >= 40:
    signal = "BUY (Liquidity Grab) 🟢"
if liq_grab_ask and confidence >= 40:
    signal = "SELL (Liquidity Grab) 🔴"
# penalize if volume divergence
if components["volume_divergence"]:
    signal += " ⚠️ (Vol Divergence)"

# ------------------------
# ---- UI / Plots ----
# ------------------------
col1, col2 = st.columns([1,2])
with col1:
    st.metric("Price", f"${current_price:,.2f}")
    st.metric("Signal", f"{signal}")
    st.metric("Confidence", f"{confidence}%")
    st.markdown("**Trends:** 1m / 5m / 15m / 1h")
    st.write(f"{'↑' if trend_1m==1 else '↓' if trend_1m==-1 else '–'}    "
             f"{'↑' if trend_5m==1 else '↓' if trend_5m==-1 else '–'}    "
             f"{'↑' if trend_15m==1 else '↓' if trend_15m==-1 else '–'}    "
             f"{'↑' if trend_1h==1 else '↓' if trend_1h==-1 else '–'}")
    st.markdown("---")
    st.write("**Key signals:**")
    st.write({
        "imbalance": round(imbalance, 4),
        "weighted_bids": round(weighted_bids, 2),
        "weighted_asks": round(weighted_asks, 2),
        "volume_spike_1m": bool(vol_spike_1m),
        "volume_recent": float(vol_recent) if vol_recent is not None else None,
        "volume_avg": float(vol_avg) if vol_avg is not None else None,
        "liquidity_grab_bid": bool(liq_grab_bid),
        "liquidity_grab_ask": bool(liq_grab_ask),
        "near_big_bid": bool(near_big_bid),
        "near_big_ask": bool(near_big_ask),
        "rsi_1m": round(float(rsi_val_1m),2)
    })
    st.caption("Confidence is computed from weighted components (orderbook, trend alignment, EMA, RSI, volume, liquidity, whales).")

with col2:
    # Candles chart (1m)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=candles_1m["time"], open=candles_1m["open"], high=candles_1m["high"],
        low=candles_1m["low"], close=candles_1m["close"], name="1m"
    ))
    # EMAs
    if len(ema_short_1m) > 0:
        fig.add_trace(go.Scatter(x=candles_1m["time"], y=ema_short_1m, name=f"EMA{ema_short_len}"))
    if len(ema_long_1m) > 0:
        fig.add_trace(go.Scatter(x=candles_1m["time"], y=ema_long_1m, name=f"EMA{ema_long_len}"))
    # Whale lines
    if big_bid is not None:
        fig.add_hline(y=float(big_bid["price"]), line_dash="dot", annotation_text=f"Big Bid {big_bid['qty']:.2f}")
    if big_ask is not None:
        fig.add_hline(y=float(big_ask["price"]), line_dash="dot", line_color="red", annotation_text=f"Big Ask {big_ask['qty']:.2f}")
    # current price
    fig.add_hline(y=current_price, line_dash="dash", line_color="orange", annotation_text="Price")
    fig.update_layout(title="BTC/USDT 1m + EMAs", xaxis_rangeslider_visible=False, height=520)
    st.plotly_chart(fig, use_container_width=True)

# Orderbook depth plot (left/right)
col3, col4 = st.columns([1,1])
with col3:
    bids["cum_qty"] = bids["qty"].cumsum()
    asks["cum_qty"] = asks["qty"].cumsum()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bids["price"], y=bids["cum_qty"], mode="lines", name="Bid Depth"))
    fig2.add_trace(go.Scatter(x=asks["price"], y=asks["cum_qty"], mode="lines", name="Ask Depth"))
    st.plotly_chart(fig2, use_container_width=True)

with col4:
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=bids["price"].astype(str), y=bids["qty"], name="Bids"))
    fig3.add_trace(go.Bar(x=asks["price"].astype(str), y=asks["qty"], name="Asks"))
    fig3.update_layout(barmode="overlay", title="Top Orderbook Levels (qty)", xaxis_title="Price", yaxis_title="Qty")
    st.plotly_chart(fig3, use_container_width=True)

# debug panel
if show_debug:
    st.markdown("### Debug / Components")
    st.json({
        "components": components,
        "confidence": confidence,
        "trends": trends,
        "weighted_bids": float(weighted_bids),
        "weighted_asks": float(weighted_asks),
        "buy_pressure": float(buy_pressure),
        "sell_pressure": float(sell_pressure),
        "big_bid": {"price": float(big_bid["price"]), "qty": float(big_bid["qty"])} if big_bid is not None else None,
        "big_ask": {"price": float(big_ask["price"]), "qty": float(big_ask["qty"])} if big_ask is not None else None,
        "vol_spike_1m": bool(vol_spike_1m),
        "volume_recent": float(vol_recent) if vol_recent is not None else None,
        "volume_avg": float(vol_avg) if vol_avg is not None else None
    })
