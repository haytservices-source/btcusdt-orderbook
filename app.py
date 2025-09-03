# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import websocket, json, threading, time
from datetime import datetime

st.set_page_config(page_title="BTC/USDT Live Order Flow", layout="wide", page_icon="📊")

SYMBOL = "BTCUSDT"
latest_data = {"bids": None, "asks": None, "last_update": None}
latest_trades = []  # will store recent trades

# ---------- Order Book WebSocket ----------
def on_message_orderbook(ws, message):
    global latest_data
    data = json.loads(message)

    if "bids" in data and "asks" in data:
        bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)

        bids["total_usd"] = bids["price"] * bids["qty"]
        asks["total_usd"] = asks["price"] * asks["qty"]

        latest_data = {
            "bids": bids,
            "asks": asks,
            "last_update": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }

def run_orderbook_ws():
    url = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth20@100ms"
    ws = websocket.WebSocketApp(url, on_message=on_message_orderbook)
    ws.run_forever()

# ---------- Trades WebSocket ----------
def on_message_trades(ws, message):
    global latest_trades
    data = json.loads(message)
    trade = {
        "price": float(data["p"]),
        "qty": float(data["q"]),
        "side": "BUY" if data["m"] is False else "SELL",  # m=False = buyer aggressive
        "time": datetime.utcfromtimestamp(data["T"]/1000).strftime("%H:%M:%S")
    }
    latest_trades.append(trade)
    # keep only last 50 trades
    latest_trades = latest_trades[-50:]

def run_trades_ws():
    url = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@trade"
    ws = websocket.WebSocketApp(url, on_message=on_message_trades)
    ws.run_forever()

# ---------- Start Threads ----------
threading.Thread(target=run_orderbook_ws, daemon=True).start()
threading.Thread(target=run_trades_ws, daemon=True).start()
time.sleep(1)

# ---------- UI ----------
st.title("📊 BTC/USDT Live Order Flow (WebSocket)")
st.caption("Streaming from Binance · Order Book + Tape (real-time)")

# Order Book
if latest_data["bids"] is None or latest_data["asks"] is None:
    st.warning("Waiting for order book data...")
    st.stop()

bids = latest_data["bids"]
asks = latest_data["asks"]

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
c1.metric("Best Bid", f"{best_bid:,.2f}")
c2.metric("Best Ask", f"{best_ask:,.2f}")
c3.metric("Spread", f"{spread:,.2f}")
c4.metric("Last Update", latest_data["last_update"])

left, right = st.columns(2)
with left:
    st.markdown("**Top Bid Walls**")
    st.dataframe(bids.nlargest(8, "total_usd")[["price", "qty", "total_usd"]], use_container_width=True)
with right:
    st.markdown("**Top Ask Walls**")
    st.dataframe(asks.nlargest(8, "total_usd")[["price", "qty", "total_usd"]], use_container_width=True)

d1, d2, d3 = st.columns(3)
d1.metric("Buy Depth (BTC)", f"{buy_btc:,.3f}")
d2.metric("Sell Depth (BTC)", f"{sell_btc:,.3f}")
d3.metric("Buy Share (%)", f"{buy_share_pct*100:,.1f}%")

if buy_share_pct >= 0.55:
    st.success("🟢 Buyers strong")
elif buy_share_pct <= 0.45:
    st.error("🔴 Sellers strong")
else:
    st.info("⚖️ Balanced")

# Depth chart
st.subheader("Depth Chart")
bids_sorted = bids.sort_values("price")
asks_sorted = asks.sort_values("price")
fig = go.Figure()
fig.add_trace(go.Bar(x=bids_sorted["price"], y=bids_sorted["qty"], name="Bids", marker_color="green", opacity=0.7))
fig.add_trace(go.Bar(x=asks_sorted["price"], y=asks_sorted["qty"], name="Asks", marker_color="red", opacity=0.7))
fig.update_layout(xaxis_title="Price", yaxis_title="BTC", height=400)
st.plotly_chart(fig, use_container_width=True)

# Trades (Tape)
st.subheader("Recent Trades (last 50)")
if latest_trades:
    trades_df = pd.DataFrame(latest_trades)
    st.dataframe(trades_df[::-1], use_container_width=True)  # newest on top

    # Delta bar chart
    buys = sum(t["qty"] for t in latest_trades if t["side"] == "BUY")
    sells = sum(t["qty"] for t in latest_trades if t["side"] == "SELL")
    delta_fig = go.Figure()
    delta_fig.add_trace(go.Bar(x=["BUY"], y=[buys], marker_color="green"))
    delta_fig.add_trace(go.Bar(x=["SELL"], y=[sells], marker_color="red"))
    delta_fig.update_layout(title="Trade Volume Delta (last 50)", height=300)
    st.plotly_chart(delta_fig, use_container_width=True)
else:
    st.info("Waiting for trades...")
