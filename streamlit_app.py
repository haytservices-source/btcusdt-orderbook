import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import websocket, json, threading, requests, time
from datetime import datetime

st.set_page_config(page_title="BTC/USDT Full Order Book", layout="wide", page_icon="📊")

SYMBOL = "BTCUSDT"
LIMIT = 100  # up to 1000 supported by REST snapshot
latest_data = {"bids": None, "asks": None, "last_update": None}

# ---------- Fetch initial snapshot ----------
def fetch_snapshot(symbol=SYMBOL, limit=LIMIT):
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    resp = requests.get(url, timeout=6)
    resp.raise_for_status()
    return resp.json()

# ---------- WebSocket Handlers ----------
def on_message(ws, message):
    global latest_data
    data = json.loads(message)

    if "b" in data and "a" in data:
        # b = bids, a = asks
        bids = pd.DataFrame(data["b"], columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(data["a"], columns=["price", "qty"], dtype=float)

        bids["total_usd"] = bids["price"] * bids["qty"]
        asks["total_usd"] = asks["price"] * asks["qty"]

        latest_data = {
            "bids": bids,
            "asks": asks,
            "last_update": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws):
    print("WebSocket closed")

def run_ws():
    url = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth@100ms"
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()

# ---------- Start WebSocket in background ----------
threading.Thread(target=run_ws, daemon=True).start()

# ---------- Load snapshot initially ----------
try:
    snapshot = fetch_snapshot()
    bids = pd.DataFrame(snapshot["bids"], columns=["price", "qty"], dtype=float)
    asks = pd.DataFrame(snapshot["asks"], columns=["price", "qty"], dtype=float)
    bids["total_usd"] = bids["price"] * bids["qty"]
    asks["total_usd"] = asks["price"] * asks["qty"]
    latest_data = {
        "bids": bids,
        "asks": asks,
        "last_update": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }
except Exception as e:
    st.error(f"Failed to load snapshot: {e}")
    st.stop()

time.sleep(1)  # give websocket time to start

# ---------- Streamlit UI ----------
st.title("📊 BTC/USDT Full Order Book (Live WebSocket + Snapshot)")
st.caption("Streaming from Binance · Up to 1000 levels")

if latest_data["bids"] is None or latest_data["asks"] is None:
    st.warning("Waiting for order book data...")
    st.stop()

bids = latest_data["bids"]
asks = latest_data["asks"]

# Metrics
best_bid = bids["price"].max()
best_ask = asks["price"].min()
spread = best_ask - best_bid

buy_btc = bids["qty"].sum()
sell_btc = asks["qty"].sum()
buy_usd = bids["total_usd"].sum()
sell_usd = asks["total_usd"].sum()
total_usd = buy_usd + sell_usd if (buy_usd + sell_usd) > 0 else 1
buy_share_pct = buy_usd / total_usd

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Bid", f"{best_bid:,.2f} USDT")
c2.metric("Best Ask", f"{best_ask:,.2f} USDT")
c3.metric("Spread", f"{spread:,.2f} USDT")
c4.metric("Last Update", latest_data["last_update"])

# Tables
left, right = st.columns(2)
with left:
    st.markdown("**Top Bid Walls (Support)**")
    st.dataframe(bids.nlargest(10, "total_usd")[["price", "qty", "total_usd"]], use_container_width=True)
with right:
    st.markdown("**Top Ask Walls (Resistance)**")
    st.dataframe(asks.nlargest(10, "total_usd")[["price", "qty", "total_usd"]], use_container_width=True)

# Depth stats
d1, d2, d3 = st.columns(3)
d1.metric("Buy Depth (BTC)", f"{buy_btc:,.3f}")
d2.metric("Sell Depth (BTC)", f"{sell_btc:,.3f}")
d3.metric("Buy Share (visible $)", f"{buy_share_pct*100:,.1f}%")

if buy_share_pct >= 0.55:
    st.success("🟢 BUYERS STRONG — visible buy-side is dominant (≥55%)")
elif buy_share_pct <= 0.45:
    st.error("🔴 SELLERS STRONG — visible sell-side is dominant (≤45%)")
else:
    st.info("⚖️ BALANCED — no clear dominance")

# Depth chart
st.subheader("Depth Chart (Amounts by Price)")
bids_sorted = bids.sort_values("price")
asks_sorted = asks.sort_values("price")

fig = go.Figure()
fig.add_trace(go.Bar(x=bids_sorted["price"], y=bids_sorted["qty"], name="Bids (Buy)", marker_color="green", opacity=0.7))
fig.add_trace(go.Bar(x=asks_sorted["price"], y=asks_sorted["qty"], name="Asks (Sell)", marker_color="red", opacity=0.7))
fig.update_layout(
    xaxis_title="Price (USDT)",
    yaxis_title="Amount (BTC)",
    legend_title_text="Side",
    bargap=0.02,
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

st.caption("Note: streaming full order book with snapshot + incremental updates. Walls can vanish instantly.")
