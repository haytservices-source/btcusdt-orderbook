# app.py — BTC/USDT Order Flow + SMC + Volume Spikes + Order Book Signals
# ---------------------------------------------------------------
# What this app adds vs your original:
# 1) Order book imbalance (existing) remains.
# 2) Recent trade flow (aggressive delta) from aggregated trades.
# 3) Volume spike detection from 1m klines (z-score style).
# 4) Trend filter using EMA(50)/EMA(200) on 1m closes.
# 5) Simple Smart Money Concepts (SMC):
#    - Liquidity sweep detection (stop hunt) on recent highs/lows.
#    - Simple BOS (break of structure) using swing points.
# 6) A combined signal score + plain-language rationale.
#
# Notes:
# - Uses Binance.US public endpoints. No API key needed.
# - Keep parameters adjustable in the sidebar to fit your style.
# - This is for education only — not financial advice.

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

# ---------------------------- Config ----------------------------
st.set_page_config(page_title="BTC/USDT — Order Flow Predictor (5s)", layout="wide", page_icon="📈")
st_autorefresh(interval=5000, key="auto_refresh")  # refresh every 5 sec

BASE = "https://api.binance.us/api/v3"
SYMBOL = "BTCUSDT"

st.title("📈 BTC/USDT — Order Flow Predictor (5s refresh)")
st.caption("Data from api.binance.us · Educational only, not financial advice")

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("Settings")
    levels = st.slider("Order book levels per side", 10, 100, 50, 10)
    show_depth = st.checkbox("Show depth chart", True)

    st.subheader("Signal Logic")
    ema_fast_len = st.number_input("EMA Fast", 10, 200, 50)
    ema_slow_len = st.number_input("EMA Slow", 20, 400, 200)
    kline_limit = st.slider("Kline history (1m bars)", 50, 1000, 300, 50)

    delta_trades = st.slider("Agg trades to read", 100, 1000, 500, 100)
    lookback_sweep = st.slider("Sweep lookback bars", 10, 100, 30, 5)
    swing_len = st.slider("Swing pivot len", 2, 10, 3)

    buy_dom_thresh = st.slider("Buy dominance % (≥)", 50, 80, 55)
    sell_dom_thresh = st.slider("Sell dominance % (≤)", 20, 50, 45)

    st.markdown("---")
    weights = {
        "orderbook": st.slider("Weight — Order Book", 0, 100, 20),
        "delta": st.slider("Weight — Tape Delta", 0, 100, 25),
        "volume": st.slider("Weight — Volume Spike", 0, 100, 20),
        "trend": st.slider("Weight — Trend (EMAs)", 0, 100, 20),
        "smc": st.slider("Weight — SMC (Sweep/BOS)", 0, 100, 15),
    }

# ---------------------------- Helpers ----------------------------
@st.cache_data(ttl=4)
def get_depth(symbol: str, limit: int):
    url = f"{BASE}/depth?symbol={symbol}&limit={limit}"
    r = requests.get(url, timeout=6)
    r.raise_for_status()
    j = r.json()
    bids = pd.DataFrame(j.get("bids", []), columns=["price", "qty"], dtype=float)
    asks = pd.DataFrame(j.get("asks", []), columns=["price", "qty"], dtype=float)
    if bids.empty or asks.empty:
        raise ValueError("Empty order book")
    bids["total_usd"] = bids.price * bids.qty
    asks["total_usd"] = asks.price * asks.qty
    return bids, asks

@st.cache_data(ttl=4)
def get_agg_trades(symbol: str, limit: int):
    # Aggregated trades include the flag 'm' (is the buyer the maker?)
    # If m == True, buyer is the maker (so SELL side was aggressive taker)
    # If m == False, buyer is the taker (so BUY side was aggressive taker)
    url = f"{BASE}/aggTrades?symbol={symbol}&limit={limit}"
    r = requests.get(url, timeout=6)
    r.raise_for_status()
    j = r.json()
    if not isinstance(j, list) or len(j) == 0:
        raise ValueError("No agg trades")
    df = pd.DataFrame(j)
    # price 'p', quantity 'q', time 'T', isBuyerMaker 'm'
    df["price"] = df["p"].astype(float)
    df["qty"] = df["q"].astype(float)
    df["time"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    df["isBuyerMaker"] = df["m"].astype(bool)
    # Taker buy volume: isBuyerMaker == False
    df["taker_buy"] = np.where(~df.isBuyerMaker, df.qty, 0.0)
    df["taker_sell"] = np.where(df.isBuyerMaker, df.qty, 0.0)
    return df

@st.cache_data(ttl=4)
def get_klines(symbol: str, interval: str = "1m", limit: int = 300):
    url = f"{BASE}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=6)
    r.raise_for_status()
    j = r.json()
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base_vol","taker_buy_quote_vol","ignore"
    ]
    df = pd.DataFrame(j, columns=cols)
    for c in ["open","high","low","close","volume","taker_buy_base_vol"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

# ---------------------------- Load Data ----------------------------
try:
    bids, asks = get_depth(SYMBOL, levels)
    trades = get_agg_trades(SYMBOL, delta_trades)
    kl = get_klines(SYMBOL, "1m", kline_limit)
    last_ok = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

# ---------------------------- Base Metrics ----------------------------
best_bid = bids.price.max()
best_ask = asks.price.min()
spread = best_ask - best_bid
mid = (best_bid + best_ask) / 2

buy_btc = bids.qty.sum()
sell_btc = asks.qty.sum()
buy_usd = (bids.price * bids.qty).sum()
sell_usd = (asks.price * asks.qty).sum()

total_usd = max(buy_usd + sell_usd, 1e-9)
buy_share = buy_usd / total_usd

# ---------------------------- Order Flow / Delta ----------------------------
# Sum taker volumes to understand aggression
buy_taker = trades.taker_buy.sum()
sell_taker = trades.taker_sell.sum()
net_delta = buy_taker - sell_taker  # >0 means buyers more aggressive

# ---------------------------- Volume Spike ----------------------------
# Compare last 1m volume vs rolling median/IQR (robust)
vol = kl.volume
last_vol = vol.iloc[-1]
med = vol.rolling(50, min_periods=10).median()
iqr = (vol.rolling(50, min_periods=10).quantile(0.75) - vol.rolling(50, min_periods=10).quantile(0.25)).replace(0, np.nan)
robust_z = (vol - med) / iqr
last_z = robust_z.iloc[-1]
vol_spike = last_z >= 2  # configurable threshold could be added

# ---------------------------- Trend (EMAs) ----------------------------
close = kl.close
ema_fast = close.ewm(span=ema_fast_len, adjust=False).mean()
ema_slow = close.ewm(span=ema_slow_len, adjust=False).mean()
trend_up = ema_fast.iloc[-1] > ema_slow.iloc[-1]
trend_down = ema_fast.iloc[-1] < ema_slow.iloc[-1]

# ---------------------------- SMC: Sweeps + BOS ----------------------------
# Liquidity sweep: price wicks beyond prior N high/low and closes back in range
hh = kl.high.rolling(lookback_sweep).max().shift(1)
ll = kl.low.rolling(lookback_sweep).min().shift(1)
last_high, last_low, last_close = kl.high.iloc[-1], kl.low.iloc[-1], kl.close.iloc[-1]

bull_sweep = (last_low < ll.iloc[-1]) and (last_close > ll.iloc[-1])
bear_sweep = (last_high > hh.iloc[-1]) and (last_close < hh.iloc[-1])

# Simple swings (pivot highs/lows)
def pivots(h, l, len_):
    ph = (h.shift(len_) < h) & (h.shift(-len_) < h)
    pl = (l.shift(len_) > l) & (l.shift(-len_) > l)
    return ph, pl

ph, pl = pivots(kl.high, kl.low, swing_len)
# last confirmed swing points
last_ph_idx = ph[::-1].idxmax()
last_pl_idx = pl[::-1].idxmax()
last_ph = kl.high.loc[last_ph_idx] if last_ph_idx != 0 else np.nan
last_pl = kl.low.loc[last_pl_idx] if last_pl_idx != 0 else np.nan

# BOS: close breaks above last swing high or below last swing low
bos_up = bool(last_close > last_ph) if not np.isnan(last_ph) else False
bos_down = bool(last_close < last_pl) if not np.isnan(last_pl) else False

# ---------------------------- Scoring ----------------------------
# Normalize components into [-1, +1], then weight to 0..100 score
comp = {}

# Order book dominance
if buy_share * 100 >= buy_dom_thresh:
    comp["orderbook"] = +1
elif buy_share * 100 <= sell_dom_thresh:
    comp["orderbook"] = -1
else:
    comp["orderbook"] = 0

# Delta (taker aggression). Scale by ratio
delta_ratio = (buy_taker - sell_taker) / max(buy_taker + sell_taker, 1e-9)
comp["delta"] = float(np.clip(delta_ratio * 2, -1, 1))  # exaggerate a bit

# Volume spike: +1 if spike with green close, -1 if spike with red close
prev_close = kl.close.iloc[-2]
bar_dir = np.sign(last_close - prev_close)  # +1 up, -1 down, 0 flat
if vol_spike:
    comp["volume"] = float(bar_dir)
else:
    comp["volume"] = 0.0

# Trend
comp["trend"] = +1 if trend_up else (-1 if trend_down else 0)

# SMC: combine sweep/BOS
smc_score = 0
if bull_sweep: smc_score += 0.5
if bear_sweep: smc_score -= 0.5
if bos_up: smc_score += 0.5
if bos_down: smc_score -= 0.5
comp["smc"] = float(np.clip(smc_score, -1, 1))

# Weighted score 0..100 (50 neutral)
wt_sum = sum(weights.values()) or 1
raw = 0.0
for k, v in comp.items():
    raw += v * (weights[k] / wt_sum)
score = int(round((raw + 1) * 50))  # map -1..+1 -> 0..100

if score > 60:
    verdict = "🟢 BUY Bias"
elif score < 40:
    verdict = "🔴 SELL Bias"
else:
    verdict = "⚖️ Neutral / Wait"

# ---------------------------- Top Tiles ----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Best Bid", f"{best_bid:,.2f}")
c2.metric("Best Ask", f"{best_ask:,.2f}")
c3.metric("Spread", f"{spread:,.2f}")
c4.metric("Buy Share (visible $)", f"{buy_share*100:,.1f}%")
c5.metric("Last Update", last_ok)

st.subheader("Signal Summary")
colA, colB = st.columns([1,2])

with colA:
    st.metric("Signal Score", f"{score}/100", help="Weighted blend of order book, delta, volume, trend, SMC")
    st.write(verdict)

    st.caption("Components (−1..+1):")
    comp_df = pd.DataFrame({"component": list(comp.keys()), "value": list(comp.values())})
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

with colB:
    # Mini price chart with EMAs and sweep markers
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=kl.open_time,
        open=kl.open, high=kl.high, low=kl.low, close=kl.close,
        name="1m"
    ))
    fig.add_trace(go.Scatter(x=kl.open_time, y=ema_fast, name=f"EMA {ema_fast_len}"))
    fig.add_trace(go.Scatter(x=kl.open_time, y=ema_slow, name=f"EMA {ema_slow_len}"))
    # markers for sweeps
    if bull_sweep:
        fig.add_trace(go.Scatter(x=[kl.open_time.iloc[-1]], y=[last_low], mode="markers+text", text=["Bull Sweep"], name="Bull Sweep"))
    if bear_sweep:
        fig.add_trace(go.Scatter(x=[kl.open_time.iloc[-1]], y=[last_high], mode="markers+text", text=["Bear Sweep"], name="Bear Sweep"))
    fig.update_layout(height=420, xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Order Book Tables ----------------------------
st.subheader("Order Book Snapshot")
left, right = st.columns(2)
with left:
    st.markdown("**Top Bid Walls (Support)**")
    top_bids = bids.nlargest(10, "total_usd")[['price','qty','total_usd']].rename(columns={"price":"Price","qty":"BTC","total_usd":"USDT"})
    st.dataframe(top_bids, use_container_width=True)
with right:
    st.markdown("**Top Ask Walls (Resistance)**")
    top_asks = asks.nlargest(10, "total_usd")[['price','qty','total_usd']].rename(columns={"price":"Price","qty":"BTC","total_usd":"USDT"})
    st.dataframe(top_asks, use_container_width=True)

# Quick metrics row
q1, q2, q3, q4 = st.columns(4)
q1.metric("Buy Depth (BTC)", f"{buy_btc:,.3f}")
q2.metric("Sell Depth (BTC)", f"{sell_btc:,.3f}")
q3.metric("Taker Buy (sum)", f"{buy_taker:,.3f}")
q4.metric("Taker Sell (sum)", f"{sell_taker:,.3f}")

# ---------------------------- Depth Chart ----------------------------
if show_depth:
    st.subheader("Depth Chart (Amounts by Price)")
    bids_sorted = bids.sort_values("price")
    asks_sorted = asks.sort_values("price")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=bids_sorted.price, y=bids_sorted.qty, name="Bids", opacity=0.7))
    fig2.add_trace(go.Bar(x=asks_sorted.price, y=asks_sorted.qty, name="Asks", opacity=0.7))
    fig2.update_layout(xaxis_title="Price (USDT)", yaxis_title="Amount (BTC)", bargap=0.02, height=380)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------- Tape Delta Bar ----------------------------
st.subheader("Aggressive Volume (Last N aggTrades)")
# Aggregate trades into 1-minute bins to visualize delta trend
tr_bins = trades.set_index("time").resample("1min").agg({"taker_buy":"sum","taker_sell":"sum"})
tr_bins["delta"] = tr_bins.taker_buy - tr_bins.taker_sell
fig3 = go.Figure()
fig3.add_trace(go.Bar(x=tr_bins.index, y=tr_bins["delta"], name="Delta"))
fig3.update_layout(height=300, xaxis_title="Time", yaxis_title="Delta (buy-sell)")
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------- Rationale / Explanation ----------------------------
reasons = []
if comp["orderbook"] > 0:
    reasons.append(f"Order book buy dominance {buy_share*100:,.1f}% ≥ {buy_dom_thresh}%")
elif comp["orderbook"] < 0:
    reasons.append(f"Order book sell dominance {100 - buy_share*100:,.1f}% ≥ {100 - sell_dom_thresh}%")

if comp["delta"] > 0:
    reasons.append("Taker flow favors buyers (positive delta)")
elif comp["delta"] < 0:
    reasons.append("Taker flow favors sellers (negative delta)")

if vol_spike:
    if bar_dir > 0:
        reasons.append("Volume spike on an up bar (demand expansion)")
    elif bar_dir < 0:
        reasons.append("Volume spike on a down bar (supply expansion)")

if trend_up:
    reasons.append("EMA trend up (fast above slow)")
elif trend_down:
    reasons.append("EMA trend down (fast below slow)")

if bull_sweep:
    reasons.append("Bullish liquidity sweep (stop hunt below prior low)")
if bear_sweep:
    reasons.append("Bearish liquidity sweep (stop hunt above prior high)")
if bos_up:
    reasons.append("BOS up: close above last swing high")
if bos_down:
    reasons.append("BOS down: close below last swing low")

st.subheader("Why this verdict?")
st.write("\n".join([f"• {r}" for r in reasons]) or "No strong factors — stay patient.")

st.caption("Note: Signals use *visible* depth + public trades/klines only. Hidden liquidity, spoofing, and fast changes can invalidate setups. Use risk management.")

