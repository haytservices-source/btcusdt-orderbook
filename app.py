# multi_exchange_orderflow_full.py
# Multi-exchange BTC/USDT order flow dashboard (Streamlit + CCXT)
# Features:
# - Fetch orderbooks + candles from multiple exchanges via CCXT
# - Normalize & aggregate orderbooks into a global view
# - Calculate Weighted Pressure Index (WPI) and a smoothed WPI (rolling)
# - Whale wall confirmation, candle confirmation, and final BUY/SELL/NEUTRAL signal
# - Optional simple simulated PnL tracker for evaluating signals
#
# Requirements:
# pip install streamlit ccxt pandas plotly numpy
# Run:
# streamlit run multi_exchange_orderflow_full.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import collections

# Try to import ccxt and provide friendly message if missing
try:
    import ccxt
except Exception as e:
    st.error("ccxt not installed. Run `pip install ccxt` in your environment and rerun this app.")
    st.stop()

# -----------------------------
# UI / Settings
# -----------------------------
st.set_page_config(page_title="Multi-Exchange Order Flow (CCXT)", layout="wide")
st.title("🌐 Multi-Exchange BTC/USDT Order Flow — Profitable Signal Engine")

st.sidebar.header("Settings")
available_exchanges = [
    ("binance", "Binance"),
    ("bybit", "Bybit"),
    ("okx", "OKX"),
    ("coinbasepro", "Coinbase Pro"),
    ("kraken", "Kraken"),
]
# map id to friendly name
ex_map = {k: v for k, v in available_exchanges}

default_selected = [k for k, _ in available_exchanges]
selected = st.sidebar.multiselect("Exchanges to include (via CCXT)", options=[k for k, _ in available_exchanges],
                                  default=default_selected)

symbol = st.sidebar.text_input("Trading pair symbol (CCXT unified)", value="BTC/USDT")
orderbook_limit = st.sidebar.selectbox("Orderbook depth levels (per exchange)", [25, 50, 100, 200], index=2)
candles_limit = st.sidebar.slider("Candles to fetch (per exchange)", 10, 200, 50)
timeframe = st.sidebar.selectbox("Candles timeframe", ["1m", "5m", "15m", "1h"], index=0)
price_round = st.sidebar.number_input("Price aggregation rounding (USD)", value=1.0, min_value=0.01, step=0.5)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 2, 30, 5)

st.sidebar.markdown("---")
show_tables = st.sidebar.checkbox("Show aggregated tables (debug)", value=False)
enable_sim = st.sidebar.checkbox("Enable simple signal PnL simulator", value=False)

# -----------------------------
# Session state (rolling buffers & simulator)
# -----------------------------
if "wpi_history" not in st.session_state:
    st.session_state.wpi_history = collections.deque(maxlen=5)

if "signals_log" not in st.session_state:
    st.session_state.signals_log = []  # store (timestamp, signal, price)

if "sim_balance" not in st.session_state:
    st.session_state.sim_balance = 10000.0  # start USD
if "sim_pos" not in st.session_state:
    st.session_state.sim_pos = 0.0  # BTC position

# -----------------------------
# Helper functions
# -----------------------------

def create_exchange_instance(id):
    try:
        ex_cls = getattr(ccxt, id)
        # set rateLimit and other params automatically
        ex = ex_cls({"enableRateLimit": True})
        return ex
    except Exception:
        return None


def safe_fetch_orderbook(ex, symbol, limit=50):
    try:
        ob = ex.fetch_order_book(symbol, limit=limit)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        return bids, asks
    except Exception:
        return None, None


def safe_fetch_ohlcv(ex, symbol, timeframe, limit=50):
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        # o = [[ts, open, high, low, close, volume], ...]
        df = pd.DataFrame(o, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception:
        return None


def round_price_fn(p):
    r = float(price_round)
    return float(np.round(p / r) * r)


# -----------------------------
# Fetch data from selected exchanges
# -----------------------------

st.sidebar.markdown("\n---\nClick 'Fetch Now' or wait for automatic refresh")
if st.sidebar.button("Fetch Now"):
    # trigger immediate refresh by sleeping briefly (will continue)
    pass

# We'll perform fetching in a single function and show spinner
with st.spinner("Fetching data from exchanges..."):
    exchange_objs = {}
    books = []  # list of (exchange_id, bids, asks)
    candles_list = []  # list of dataframes with exchange column

    for ex_id in selected:
        ex = create_exchange_instance(ex_id)
        if ex is None:
            continue
        # Some exchanges require symbol mapping. We'll try provided symbol, then try BTC/USDT or BTC/USD fallback
        bids, asks = safe_fetch_orderbook(ex, symbol, limit=orderbook_limit)
        if bids is None or asks is None or len(bids) == 0 or len(asks) == 0:
            # try common fallbacks
            for alt in ["BTC/USDT", "BTC/USD", "BTC/USDT:USDT"]:
                if alt == symbol:
                    continue
                bids, asks = safe_fetch_orderbook(ex, alt, limit=orderbook_limit)
                if bids and asks:
                    break
        if bids and asks:
            books.append((ex_id, bids, asks))

        # candles
        dfc = safe_fetch_ohlcv(ex, symbol, timeframe, limit=candles_limit)
        if dfc is None:
            # try fallbacks
            for alt in ["BTC/USDT", "BTC/USD"]:
                if alt == symbol:
                    continue
                dfc = safe_fetch_ohlcv(ex, alt, timeframe, limit=candles_limit)
                if dfc is not None:
                    break
        if dfc is not None:
            dfc["exchange"] = ex_id
            candles_list.append(dfc)

    # If all books empty, try common public/alternate endpoints via simple REST (lightweight) - optional
    # (omitted — CCXT covers most)

# -----------------------------
# Aggregate orderbooks
# -----------------------------

if not books:
    st.error("No orderbook data could be fetched from the selected exchanges. Try selecting different exchanges or check network.")
    st.stop()

# Build combined bids/asks (aggregate by rounded price)
all_bids = []
all_asks = []
for ex_id, bids, asks in books:
    for price, qty in bids:
        all_bids.append({"price": float(price), "qty": float(qty), "exchange": ex_id})
    for price, qty in asks:
        all_asks.append({"price": float(price), "qty": float(qty), "exchange": ex_id})

if not all_bids or not all_asks:
    st.error("Aggregated book is empty after fetching.")
    st.stop()

df_bids = pd.DataFrame(all_bids)
df_asks = pd.DataFrame(all_asks)

df_bids["price_r"] = df_bids["price"].apply(round_price_fn)
df_asks["price_r"] = df_asks["price"].apply(round_price_fn)

agg_bids = df_bids.groupby("price_r", as_index=False).agg({"qty": "sum"}).rename(columns={"price_r": "price"})
agg_asks = df_asks.groupby("price_r", as_index=False).agg({"qty": "sum"}).rename(columns={"price_r": "price"})

agg_bids = agg_bids.sort_values("price", ascending=False).reset_index(drop=True)
agg_asks = agg_asks.sort_values("price", ascending=True).reset_index(drop=True)

# -----------------------------
# Compute mid price
# -----------------------------
if not agg_bids.empty and not agg_asks.empty:
    mid_price = (agg_bids["price"].max() + agg_asks["price"].min()) / 2.0
else:
    mid_price = None

# -----------------------------
# Weighted Pressure Index (WPI)
# -----------------------------

if mid_price is None:
    st.error("Cannot determine mid price for aggregated book.")
    st.stop()

# compute weight: qty / distance
agg_bids["weight"] = agg_bids.apply(lambda r: r["qty"] / max(1e-8, max(0.0001, mid_price - r["price"])), axis=1)
agg_asks["weight"] = agg_asks.apply(lambda r: r["qty"] / max(1e-8, max(0.0001, r["price"] - mid_price)), axis=1)

weighted_bids = agg_bids["weight"].sum()
weighted_asks = agg_asks["weight"].sum()

wpi = (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks) if (weighted_bids + weighted_asks) > 0 else 0.0

# Smoothed WPI (rolling)
st.session_state.wpi_history.append(wpi)
smoothed_wpi = float(sum(st.session_state.wpi_history) / len(st.session_state.wpi_history))

# -----------------------------
# Whale detection (combined book)
# -----------------------------
big_bid_row = agg_bids.loc[agg_bids["qty"].idxmax()] if not agg_bids.empty else None
big_ask_row = agg_asks.loc[agg_asks["qty"].idxmax()] if not agg_asks.empty else None

nearest_bid_row = agg_bids[agg_bids["price"] < mid_price].sort_values("price", ascending=False).head(1)
nearest_ask_row = agg_asks[agg_asks["price"] > mid_price].sort_values("price", ascending=True).head(1)

if not nearest_bid_row.empty and not nearest_ask_row.empty:
    nb_price, nb_qty = float(nearest_bid_row.iloc[0]["price"]), float(nearest_bid_row.iloc[0]["qty"])
    na_price, na_qty = float(nearest_ask_row.iloc[0]["price"]), float(nearest_ask_row.iloc[0]["qty"])
else:
    nb_price = na_price = nb_qty = na_qty = None

median_order_size = pd.concat([agg_bids["qty"], agg_asks["qty"]]).median()
strong_buy_wall = nb_qty is not None and nb_qty > 2 * max(1e-8, median_order_size)
strong_sell_wall = na_qty is not None and na_qty > 2 * max(1e-8, median_order_size)

# -----------------------------
# Candle confirmation (aggregate candles)
# -----------------------------

def combine_candles(list_of_dfs):
    if not list_of_dfs:
        return None
    df = pd.concat(list_of_dfs, ignore_index=True)
    df = df.dropna(subset=["time"]).copy()
    # group by time (timestamp) and aggregate
    dfg = df.groupby("time").agg({
        "open": "mean",
        "high": "max",
        "low": "min",
        "close": "mean",
        "volume": "sum"
    }).reset_index().sort_values("time")
    return dfg

combined_candles = combine_candles(candles_list)

candle_trend = None
if combined_candles is not None and len(combined_candles) >= 2:
    last = combined_candles.iloc[-1]["close"]
    prev = combined_candles.iloc[-2]["close"]
    candle_trend = "bullish" if last > prev else "bearish"

# -----------------------------
# Trading signal logic
# -----------------------------

# thresholds
wpi_threshold = st.sidebar.slider("WPI threshold (abs)", 0.1, 0.6, 0.35, step=0.05)

signal = "⚖️ Neutral / Wait"
if smoothed_wpi > wpi_threshold and strong_buy_wall and candle_trend == "bullish":
    signal = "✅ BUY"
elif smoothed_wpi < -wpi_threshold and strong_sell_wall and candle_trend == "bearish":
    signal = "❌ SELL"
else:
    signal = "⚖️ Neutral / Wait"

# -----------------------------
# Simple simulator (very basic) — optional
# -----------------------------
if enable_sim and signal in ("✅ BUY", "❌ SELL"):
    # simulate market fill at mid_price and simple exit after opposite signal
    now_price = mid_price
    last_signal = st.session_state.signals_log[-1] if st.session_state.signals_log else None
    # Append current signal
    st.session_state.signals_log.append((pd.Timestamp.now(), signal, now_price))
    # Very naive position sizing: 10% of balance
    size_usd = st.session_state.sim_balance * 0.1
    size_btc = size_usd / now_price
    if signal == "✅ BUY":
        st.session_state.sim_balance -= size_usd
        st.session_state.sim_pos += size_btc
    else:
        # SELL -> close equivalent amount from position if available
        sell_amt = min(st.session_state.sim_pos, size_btc)
        st.session_state.sim_balance += sell_amt * now_price
        st.session_state.sim_pos -= sell_amt

# show simulated PnL if enabled
if enable_sim:
    current_value = st.session_state.sim_balance + st.session_state.sim_pos * mid_price
    st.sidebar.metric("Simulated Account Value (USD)", f"${current_value:,.2f}")
    st.sidebar.metric("Position (BTC)", f"{st.session_state.sim_pos:.6f}")

# -----------------------------
# Visuals
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Aggregated Mid Price", f"${mid_price:,.2f}")
col2.metric("Weighted Buy Pressure", f"{weighted_bids:,.2f}")
col3.metric("Weighted Sell Pressure", f"{weighted_asks:,.2f}")

st.subheader(f"Market Signal → {signal}  |  Smoothed WPI: {smoothed_wpi:.3f}")
st.caption(f"Sources: {', '.join([ex_map.get(e, e) for e, _, _ in books]) if books else ', '.join(selected)}")

if big_bid_row is not None and big_ask_row is not None:
    st.caption(f"🐋 Biggest Buy Wall: {big_bid_row['qty']:.2f} @ ${big_bid_row['price']:.2f} | "
               f"🐋 Biggest Sell Wall: {big_ask_row['qty']:.2f} @ ${big_ask_row['price']:.2f}")

# Left: depth chart, Right: candles
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🔥 Aggregated Depth (Cumulative)")
    bids_cum = agg_bids.copy()
    bids_cum["cum_qty"] = bids_cum["qty"].cumsum()
    asks_cum = agg_asks.copy()
    asks_cum["cum_qty"] = asks_cum["qty"].cumsum()

    fig_depth = go.Figure()
    fig_depth.add_trace(go.Scatter(x=bids_cum["price"], y=bids_cum["cum_qty"], fill="tozeroy", name="Bids (cum)", mode="lines"))
    fig_depth.add_trace(go.Scatter(x=asks_cum["price"], y=asks_cum["cum_qty"], fill="tozeroy", name="Asks (cum)", mode="lines"))
    fig_depth.add_vline(x=mid_price, line_dash="dot", annotation_text="Mid Price")
    fig_depth.update_layout(xaxis_title="Price", yaxis_title="Cumulative Qty", height=600)
    st.plotly_chart(fig_depth, use_container_width=True)

with col_right:
    st.markdown("### 🕯️ Aggregated Candles")
    if combined_candles is None:
        st.info("No aggregated candles available.")
    else:
        fig_c = go.Figure(data=[go.Candlestick(
            x=combined_candles["time"], open=combined_candles["open"], high=combined_candles["high"],
            low=combined_candles["low"], close=combined_candles["close"], name="Price")])
        fig_c.add_hline(y=mid_price, line_dash="dot", annotation_text="Mid Price")
        fig_c.update_layout(height=600)
        st.plotly_chart(fig_c, use_container_width=True)

# Debug tables
if show_tables:
    with st.expander("Aggregated Book (top 100)"):
        st.write("Top bids:")
        st.dataframe(agg_bids.head(100))
        st.write("Top asks:")
        st.dataframe(agg_asks.head(100))

    with st.expander("Raw fetched candles per exchange (sample)"):
        if candles_list:
            for c in candles_list:
                st.write(c["exchange"].iloc[0] if "exchange" in c.columns else "unknown")
                st.dataframe(c.head(5))
        else:
            st.write("No candle data fetched.")

# Signals log
with st.expander("Signals log (recent)"):
    if st.session_state.signals_log:
        df_log = pd.DataFrame(st.session_state.signals_log, columns=["timestamp", "signal", "price"]).sort_values("timestamp", ascending=False)
        st.dataframe(df_log.head(50))
    else:
        st.write("No signals logged yet.")

# final tips
st.markdown("---")
st.markdown(
    "**Tips:**\n"
    "- Adjust WPI threshold and rolling window size for your desired sensitivity.\n"
    "- Use the simulator toggle to get a rough feel for how signals would behave (very naive).\n"
    "- For production, consider running this on a VPS and using exchange API keys/WS endpoints for lower latency.\n"
)

# auto-refresh sleep
# We wait refresh_seconds so Streamlit's rerun isn't too aggressive
time.sleep(refresh_seconds)
