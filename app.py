# multi_exchange_orderflow.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- Auto-refresh every 2 seconds ---
st_autorefresh(interval=2000, key="refresh")

st.set_page_config(page_title="Multi-Exchange BTC/USDT Order Flow", layout="wide")
st.title("🌐 Multi-Exchange BTC/USDT Order Flow (Aggregated)")

# ---------- UI Controls ----------
st.sidebar.header("Settings")
available_exchanges = ["Binance.US", "Bybit", "OKX", "Coinbase"]
selected_exchanges = st.sidebar.multiselect("Select exchanges to include", available_exchanges,
                                            default=available_exchanges)
timeframe = st.sidebar.selectbox("Candles timeframe", ["1m", "5m", "15m", "1h"], index=0)
candles_limit = st.sidebar.slider("Candles to fetch (per exchange)", 50, 500, 100, step=10)
orderbook_limit = st.sidebar.selectbox("Orderbook depth levels", [50, 100, 200], index=2)
price_round = st.sidebar.number_input("Price aggregation rounding (USD)", value=1.0, min_value=0.01, step=0.5)
st.sidebar.markdown("⚠️ Some exchanges may block requests from your IP. App will still show aggregated results from the available sources.")


# ---------- Helper utilities ----------
def safe_request_get(url, params=None, headers=None, timeout=4):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        return None

def round_price(p):
    # round price to nearest `price_round`
    r = float(price_round)
    return float(np.round(p / r) * r)

# ---------- Exchange-specific fetchers ----------
def fetch_orderbook_binance_us(limit=200):
    url = "https://api.binance.us/api/v3/depth"
    params = {"symbol": "BTCUSDT", "limit": limit}
    r = safe_request_get(url, params=params)
    if not r:
        return None
    j = r.json()
    bids = pd.DataFrame(j.get("bids", []), columns=["price", "qty"], dtype=float)
    asks = pd.DataFrame(j.get("asks", []), columns=["price", "qty"], dtype=float)
    bids["exchange"] = "Binance.US"
    asks["exchange"] = "Binance.US"
    return bids, asks

def fetch_orderbook_bybit(limit=200):
    # Bybit v5 public orderbook
    url = "https://api.bybit.com/v5/market/orderbook"
    params = {"symbol": "BTCUSDT", "limit": limit}
    r = safe_request_get(url, params=params)
    if not r:
        return None
    j = r.json()
    try:
        data = j.get("result", {}).get("data", [])
        if isinstance(data, list) and len(data) > 0:
            # Bybit returns nested asks/bids - pick first data item
            item = data[0]
            bids = pd.DataFrame(item.get("bids", []), columns=["price", "qty"], dtype=float)
            asks = pd.DataFrame(item.get("asks", []), columns=["price", "qty"], dtype=float)
        else:
            # older format fallback
            bids = pd.DataFrame(j.get("result", {}).get("b", []), columns=["price", "qty"], dtype=float)
            asks = pd.DataFrame(j.get("result", {}).get("a", []), columns=["price", "qty"], dtype=float)
    except Exception:
        return None
    bids["exchange"] = "Bybit"
    asks["exchange"] = "Bybit"
    return bids, asks

def fetch_orderbook_okx(limit=200):
    # OKX public orderbook
    url = "https://www.okx.com/api/v5/market/books"
    params = {"instId": "BTC-USDT", "sz": limit}
    r = safe_request_get(url, params=params)
    if not r:
        return None
    j = r.json()
    try:
        data = j.get("data", [])[0]
        asks = pd.DataFrame(data.get("asks", []), columns=["price", "qty", "something"], dtype=float)[["price", "qty"]]
        bids = pd.DataFrame(data.get("bids", []), columns=["price", "qty", "something"], dtype=float)[["price", "qty"]]
    except Exception:
        return None
    bids["exchange"] = "OKX"
    asks["exchange"] = "OKX"
    return bids, asks

def fetch_orderbook_coinbase(limit=200):
    # Coinbase book level 2
    url = "https://api.exchange.coinbase.com/products/BTC-USD/book"
    params = {"level": 2}
    r = safe_request_get(url, params=params)
    if not r:
        return None
    j = r.json()
    bids = pd.DataFrame(j.get("bids", []), columns=["price", "qty", "num"], dtype=float)[["price", "qty"]]
    asks = pd.DataFrame(j.get("asks", []), columns=["price", "qty", "num"], dtype=float)[["price", "qty"]]
    bids["exchange"] = "Coinbase"
    asks["exchange"] = "Coinbase"
    # Coinbase prices are USD; treat USD ~= USDT
    return bids, asks

# ---------- Candles fetchers ----------
def fetch_candles_binance_us(interval="1m", limit=100):
    url = "https://api.binance.us/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    r = safe_request_get(url, params=params)
    if not r:
        return None
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades","taker_base","taker_quote","ignore"
    ])
    df["time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms")
    df = df[["time","open","high","low","close","volume"]].astype(float)
    return df

def fetch_candles_bybit(interval="1m", limit=100):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    r = safe_request_get(url, params=params)
    if not r:
        return None
    j = r.json()
    try:
        lst = j.get("result", {}).get("list", [])
        df = pd.DataFrame(lst, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"].astype(float), unit="s")
        return df.astype(float)
    except Exception:
        return None

def fetch_candles_okx(interval="1m", limit=100):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": "BTC-USDT", "bar": interval, "limit": limit}
    r = safe_request_get(url, params=params)
    if not r:
        return None
    j = r.json()
    try:
        # OKX returns list of [ts, open, high, low, close, volume]
        raw = j.get("data", [])
        df = pd.DataFrame(raw, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"].astype(float), unit="ms")
        return df.astype(float)
    except Exception:
        return None

def fetch_candles_coinbase(granularity=60, limit=100):
    # Coinbase uses granularity in seconds; 1m = 60
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    params = {"granularity": granularity}
    r = safe_request_get(url, params=params)
    if not r:
        return None
    raw = r.json()
    # raw: [time, low, high, open, close, volume]
    df = pd.DataFrame(raw, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"].astype(float), unit="s")
    df = df[["time","open","high","low","close","volume"]]
    return df.astype(float)

# ---------- Aggregate fetch helpers ----------
def fetch_selected_orderbooks(exchanges, limit=200):
    books = []
    for ex in exchanges:
        try:
            if ex == "Binance.US":
                out = fetch_orderbook_binance_us(limit=limit)
            elif ex == "Bybit":
                out = fetch_orderbook_bybit(limit=limit)
            elif ex == "OKX":
                out = fetch_orderbook_okx(limit=limit)
            elif ex == "Coinbase":
                out = fetch_orderbook_coinbase(limit=limit)
            else:
                out = None
            if out:
                bids, asks = out
                books.append((bids, asks))
        except Exception:
            continue
    return books

def fetch_selected_candles(exchanges, interval="1m", limit=100):
    candles = []
    # map timeframe to exchange-specific params
    # Coinbase granularity mapping
    gran_map = {"1m":60, "5m":300, "15m":900, "1h":3600}
    for ex in exchanges:
        try:
            if ex == "Binance.US":
                c = fetch_candles_binance_us(interval=interval, limit=limit)
            elif ex == "Bybit":
                c = fetch_candles_bybit(interval=interval, limit=limit)
            elif ex == "OKX":
                c = fetch_candles_okx(interval=interval, limit=limit)
            elif ex == "Coinbase":
                c = fetch_candles_coinbase(granularity=gran_map.get(interval,60), limit=limit)
            else:
                c = None
            if c is not None and not c.empty:
                c["exchange"] = ex
                candles.append(c)
        except Exception:
            continue
    return candles

# ---------- Build combined orderbook ----------
def build_combined_orderbook(books):
    # books: list of (bids, asks)
    all_bids = []
    all_asks = []
    for bids, asks in books:
        if bids is not None and not bids.empty:
            bids2 = bids.copy()
            bids2["price_r"] = bids2["price"].apply(round_price)
            all_bids.append(bids2[["price_r","qty","exchange"]])
        if asks is not None and not asks.empty:
            asks2 = asks.copy()
            asks2["price_r"] = asks2["price"].apply(round_price)
            all_asks.append(asks2[["price_r","qty","exchange"]])
    if not all_bids and not all_asks:
        return None, None
    df_bids = pd.concat(all_bids, ignore_index=True) if all_bids else pd.DataFrame(columns=["price_r","qty","exchange"])
    df_asks = pd.concat(all_asks, ignore_index=True) if all_asks else pd.DataFrame(columns=["price_r","qty","exchange"])
    # group by rounded price and sum qty
    df_bids = df_bids.groupby("price_r", as_index=False).agg({"qty":"sum"}).rename(columns={"price_r":"price"})
    df_asks = df_asks.groupby("price_r", as_index=False).agg({"qty":"sum"}).rename(columns={"price_r":"price"})
    # convert to numeric, sort
    df_bids = df_bids.sort_values("price", ascending=False).reset_index(drop=True)
    df_asks = df_asks.sort_values("price", ascending=True).reset_index(drop=True)
    return df_bids, df_asks

# ---------- Combine candles (time-align & average) ----------
def build_combined_candles(candles_list):
    # candles_list: list of dataframes with columns time, open,high,low,close,volume,exchange
    if not candles_list:
        return None
    # Concatenate and pivot by time
    df = pd.concat(candles_list, ignore_index=True)
    # Round time to the timeframe resolution by taking the timestamp as-is (exchanges should align)
    df = df.dropna(subset=["time"])
    # Group by time and compute mean of OHLC, sum of volume
    agg = df.groupby("time").agg({
        "open": "mean",
        "high": "max",   # conservative
        "low": "min",
        "close": "mean",
        "volume": "sum"
    }).reset_index().sort_values("time")
    return agg

# ---------- Main logic ----------
with st.spinner("Fetching data from selected exchanges..."):
    books = fetch_selected_orderbooks(selected_exchanges, limit=orderbook_limit)
    candles_raw = fetch_selected_candles(selected_exchanges, interval=timeframe, limit=candles_limit)

if not books:
    st.error("Failed to fetch orderbook data from all selected exchanges. Try enabling more exchanges or check your network.")
    # continue to allow candles if available

combined_bids, combined_asks = build_combined_orderbook(books)

combined_candles = build_combined_candles(candles_raw)

# If both combined books and candles are missing, show error and stop
if (combined_bids is None or combined_asks is None or combined_bids.empty or combined_asks.empty) and (combined_candles is None or combined_candles.empty):
    st.error("No usable data fetched from selected exchanges.")
    st.stop()

# --- Compute mid price from combined orderbook if available, else from last candle close
if combined_bids is not None and not combined_bids.empty and combined_asks is not None and not combined_asks.empty:
    mid_price = (combined_bids["price"].max() + combined_asks["price"].min()) / 2.0
else:
    # fallback to last known candle close
    if combined_candles is not None and not combined_candles.empty:
        mid_price = float(combined_candles.iloc[-1]["close"])
    else:
        mid_price = 0.0

# --- Weighted pressure calculation on combined book ---
if combined_bids is not None and not combined_bids.empty:
    combined_bids = combined_bids.copy()
    combined_bids["weight"] = combined_bids.apply(lambda r: r["qty"] / max(1e-6, max(0.0001, mid_price - r["price"])), axis=1)
else:
    combined_bids = pd.DataFrame(columns=["price","qty","weight"])

if combined_asks is not None and not combined_asks.empty:
    combined_asks = combined_asks.copy()
    combined_asks["weight"] = combined_asks.apply(lambda r: r["qty"] / max(1e-6, max(0.0001, r["price"] - mid_price)), axis=1)
else:
    combined_asks = pd.DataFrame(columns=["price","qty","weight"])

weighted_bids = combined_bids["weight"].sum() if not combined_bids.empty else 0.0
weighted_asks = combined_asks["weight"].sum() if not combined_asks.empty else 0.0

wpi = (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks) if (weighted_bids + weighted_asks) > 0 else 0.0

# --- Bias and projection ---
if wpi > 0.25:
    bias = "🔥 Buyers Dominant (Bullish)"
    confidence = f"{wpi*100:.1f}%"
elif wpi < -0.25:
    bias = "🔴 Sellers Dominant (Bearish)"
    confidence = f"{abs(wpi)*100:.1f}%"
else:
    bias = "⚖️ Neutral / Sideways"
    confidence = f"{abs(wpi)*100:.1f}%"

# Whale detection: biggest levels in combined book
big_bid = combined_bids.loc[combined_bids["qty"].idxmax()] if not combined_bids.empty else None
big_ask = combined_asks.loc[combined_asks["qty"].idxmax()] if not combined_asks.empty else None

nearest_bid = combined_bids[combined_bids["price"] < mid_price].sort_values("price", ascending=False).head(1)
nearest_ask = combined_asks[combined_asks["price"] > mid_price].sort_values("price", ascending=True).head(1)

if not nearest_bid.empty and not nearest_ask.empty:
    nb_price, nb_qty = float(nearest_bid.iloc[0]["price"]), float(nearest_bid.iloc[0]["qty"])
    na_price, na_qty = float(nearest_ask.iloc[0]["price"]), float(nearest_ask.iloc[0]["qty"])
else:
    nb_price = na_price = nb_qty = na_qty = None

if nb_qty is not None and na_qty is not None:
    if nb_qty > na_qty and wpi > 0:
        projection = f"📈 Likely Upward → Next Zone ${na_price:,.0f}"
    elif na_qty > nb_qty and wpi < 0:
        projection = f"📉 Likely Downward → Next Zone ${nb_price:,.0f}"
    else:
        projection = f"🤔 Unclear → Range ${nb_price:,.0f} - ${na_price:,.0f}"
else:
    projection = "No clear projection (insufficient depth around mid price)."

# ---------- Display metrics ----------
col1, col2, col3 = st.columns(3)
col1.metric("Combined Mid Price", f"${mid_price:,.2f}")
col2.metric("Weighted Buy Pressure", f"{weighted_bids:,.2f}")
col3.metric("Weighted Sell Pressure", f"{weighted_asks:,.2f}")

st.subheader(f"Market Bias → {bias} | Confidence: {confidence}")
st.subheader(f"Projection → {projection}")

if big_bid is not None and big_ask is not None:
    st.caption(f"🐋 Biggest Buy Wall: {big_bid['qty']:.2f} BTC @ ${big_bid['price']:.0f} | "
               f"🐋 Biggest Sell Wall: {big_ask['qty']:.2f} BTC @ ${big_ask['price']:.0f}")

# ---------- Layout: Depth (left) and Candles (right) ----------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🔥 Aggregated Order Book Depth")
    if combined_bids is None or combined_bids.empty or combined_asks is None or combined_asks.empty:
        st.info("Not enough aggregated book data to display depth chart.")
    else:
        # cumulative depth
        bids_cum = combined_bids.copy()
        bids_cum["cum_qty"] = bids_cum["qty"].cumsum()
        asks_cum = combined_asks.copy()
        asks_cum["cum_qty"] = asks_cum["qty"].cumsum()

        fig_depth = go.Figure()
        fig_depth.add_trace(go.Scatter(x=bids_cum["price"], y=bids_cum["cum_qty"], fill="tozeroy", name="Bids (cum)", mode="lines"))
        fig_depth.add_trace(go.Scatter(x=asks_cum["price"], y=asks_cum["cum_qty"], fill="tozeroy", name="Asks (cum)", mode="lines"))
        fig_depth.update_layout(title="Aggregated Depth Chart (cumulative)", xaxis_title="Price", yaxis_title="Cumulative Quantity", height=600)
        fig_depth.add_vline(x=mid_price, line_dash="dot", annotation_text="Mid Price", annotation_position="top right")
        st.plotly_chart(fig_depth, use_container_width=True)

with col_right:
    st.markdown("### 🕯️ Aggregated Candles")
    if combined_candles is None or combined_candles.empty:
        st.info("No aggregated candles available from selected exchanges.")
    else:
        fig_c = go.Figure(data=[go.Candlestick(
            x=combined_candles["time"],
            open=combined_candles["open"],
            high=combined_candles["high"],
            low=combined_candles["low"],
            close=combined_candles["close"],
            name="Price"
        )])
        fig_c.update_layout(title=f"Aggregated BTC Candles ({timeframe})", xaxis_title="Time", yaxis_title="Price", height=600)
        fig_c.add_hline(y=mid_price, line_dash="dot", annotation_text="Mid Price")
        st.plotly_chart(fig_c, use_container_width=True)

# ---------- Optional: Show tables for debugging ----------
with st.expander("Show combined top-of-book (debug)"):
    st.write("Top bids (aggregated):")
    if combined_bids is not None and not combined_bids.empty:
        st.dataframe(combined_bids.head(50))
    else:
        st.write("No bids.")

    st.write("Top asks (aggregated):")
    if combined_asks is not None and not combined_asks.empty:
        st.dataframe(combined_asks.head(50))
    else:
        st.write("No asks.")

with st.expander("Raw candles used for aggregation (per exchange)"):
    if candles_raw:
        for c in candles_raw:
            st.write(c["exchange"].iloc[0] if "exchange" in c.columns else "unknown", c.head(3))
    else:
        st.write("No raw candles fetched.")

# ---------- Final note ----------
st.markdown(
    """
    **Notes & tips**
    - This app aggregates public orderbooks across exchanges; some exchanges may block requests from certain regions/IPs.
    - If an exchange fails, the app continues with the other enabled exchanges.
    - Price rounding (in left sidebar) controls how price levels are grouped when aggregating across exchanges.
    - For production or heavy use, consider a paid market-data provider or run this behind a proxy/VPN if a given exchange blocks requests from your region.
    """
)
