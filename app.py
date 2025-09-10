# ml_orderflow_predictor.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import math
import plotly.graph_objects as go

st.set_page_config(page_title="ML Orderflow Predictor (Honest)", layout="wide")
st.title("📈 ML Orderflow Predictor — realistic (No 100% guarantees)")

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Symbol (try BTC/USDT)", "BTC/USDT")
timeframe = st.sidebar.selectbox("Candles timeframe", ["1m","5m","15m","1h"], index=0)
history_bars = st.sidebar.slider("History bars for training", 100, 2000, 500, step=50)
test_size = st.sidebar.slider("Backtest test split (%)", 10, 50, 20)
use_sklearn = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
except Exception:
    use_sklearn = False

st.sidebar.markdown("---")
st.sidebar.write("Tip: This app shows *probabilistic* predictions and backtest metrics.")
st.sidebar.write("Install scikit-learn for an ML model (pip install scikit-learn).")

# -------------------------------------------------------------------
# Utility: fetch candles from multiple public endpoints (try Bybit, OKX, Binance.US)
# -------------------------------------------------------------------
def fetch_candles_bybit(interval="1m", limit=500):
    try:
        # Bybit expects interval like '1' for min '1' but v5 kline uses '1' etc. We'll use v2 public kline if available
        url = "https://api.bybit.com/v2/public/kline/list"
        params = {"symbol": "BTCUSDT", "interval": interval.replace("m",""), "limit": limit}
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        if j.get("ret_code") == 0 and j.get("result"):
            lst = j["result"]
            df = pd.DataFrame(lst)
            df = df.rename(columns={"open_time":"time","open":"open","high":"high","low":"low","close":"close","volume":"volume"})
            df["time"] = pd.to_datetime(df["time"], unit="s")
            return df[["time","open","high","low","close","volume"]].astype(float)
    except Exception:
        pass
    return None

def fetch_candles_okx(interval="1m", limit=500):
    try:
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId":"BTC-USDT", "bar": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=6)
        j = r.json()
        data = j.get("data")
        if data:
            df = pd.DataFrame(data, columns=["time","open","high","low","close","volume","x","y"])
            df["time"] = pd.to_datetime(df["time"].astype(float), unit="ms")
            df = df[["time","open","high","low","close","volume"]].astype(float)
            return df
    except Exception:
        pass
    return None

def fetch_candles_binance_us(interval="1m", limit=500):
    try:
        url = "https://api.binance.us/api/v3/klines"
        params = {"symbol":"BTCUSDT", "interval":interval, "limit":limit}
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        df = pd.DataFrame(j, columns=[
            "open_time","open","high","low","close","volume","close_time","a","b","c","d","e"
        ])
        df["time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms")
        df = df[["time","open","high","low","close","volume"]].astype(float)
        return df
    except Exception:
        pass
    return None

# Try multiple providers and pick the first that works (or combine)
candles = None
providers = [fetch_candles_bybit, fetch_candles_okx, fetch_candles_binance_us]
for p in providers:
    df = p(interval=timeframe, limit=history_bars+50)
    if df is not None and len(df) >= history_bars:
        candles = df.tail(history_bars).reset_index(drop=True)
        break

if candles is None:
    st.error("Failed to fetch sufficient candles from Bybit/OKX/Binance.US. Check network or try a different timeframe.")
    st.stop()

st.success(f"Fetched {len(candles)} candles from provider (timeframe {timeframe}).")

# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------
df = candles.copy()
df["return"] = df["close"].pct_change().fillna(0)
df["logret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
df["ma5"] = df["close"].rolling(5, min_periods=1).mean()
df["ma10"] = df["close"].rolling(10, min_periods=1).mean()
df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
df["ma5_ma20"] = (df["ma5"] - df["ma20"]) / df["ma20"]
df["vol_roc"] = df["volume"].pct_change().fillna(0)
df["range"] = (df["high"] - df["low"]) / df["open"]
# simple order-flow proxy: (close-open)/range if range>0
df["of_proxy"] = np.where(df["range"]>0, (df["close"] - df["open"]) / df["range"], 0)

# Label: next candle up (1) vs down (0)
df["next_close"] = df["close"].shift(-1)
df["target"] = (df["next_close"] > df["close"]).astype(int)
df = df.dropna().reset_index(drop=True)

features = ["return","ma5_ma20","vol_roc","of_proxy","range"]
X = df[features].values
y = df["target"].values

# -------------------------------------------------------------------
# Train / Test split (time-ordered)
# -------------------------------------------------------------------
split_idx = int(len(X) * (1 - test_size/100.0))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

# -------------------------------------------------------------------
# Model (logistic regression if available, else simple rule)
# -------------------------------------------------------------------
model = None
if use_sklearn:
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Model: Logistic Regression (sklearn)")
        st.write(f"Accuracy on test set: {acc:.4f}")
        st.write(pd.DataFrame(report).transpose())
        st.write("Confusion matrix (rows: true, cols: pred)")
        st.write(cm)
    except Exception as e:
        st.warning("sklearn model failed to train; falling back to simple rule. Error: " + str(e))
        use_sklearn = False

if not use_sklearn:
    # deterministic fallback: if ma5 > ma20 and vol spike then predict up
    def rule_predict(Xarr):
        preds = []
        for row in Xarr:
            ma_diff = row[1]
            vol_roc = row[2]
            ofp = row[3]
            score = 0
            score += 1 if ma_diff > 0 else -1
            score += 1 if vol_roc > 0.5 else 0
            score += 1 if ofp > 0.2 else 0
            preds.append(1 if score > 0 else 0)
        return np.array(preds)
    y_pred = rule_predict(X_test)
    y_prob = None
    acc = (y_pred == y_test).mean()
    st.subheader("Model: Deterministic Rule (fallback)")
    st.write(f"Rule accuracy on test set: {acc:.4f}")
    st.write("Rule: ma5>ma20 + volume spike + strong close/open proportion -> predict UP")

# -------------------------------------------------------------------
# Backtest a naive strategy that trades every prediction (very naive)
# -------------------------------------------------------------------
st.subheader("Backtest (naive execution at next open price) — illustrative only")
initial_cash = 10000.0
cash = initial_cash
btc = 0.0
trade_log = []
test_prices = df_test["close"].values  # price we assume for entry
for i, pred in enumerate(y_pred):
    price = test_prices[i]
    # naive position sizing: use 10% cash on BUY, close 10% on SELL
    size_usd = initial_cash * 0.05
    if pred == 1:
        # buy fraction
        qty = size_usd / price
        # execute
        cash -= size_usd
        btc += qty
        trade_log.append(("BUY", df_test["time"].iloc[i], price, qty))
    else:
        # sell small portion if we have
        qty = min(btc, size_usd / price)
        cash += qty * price
        btc -= qty
        trade_log.append(("SELL", df_test["time"].iloc[i], price, qty))
# final portfolio value
final_value = cash + btc * test_prices[-1]
st.write(f"Initial: ${initial_cash:.2f} → Final (naive): ${final_value:.2f} (BTC position {btc:.6f})")
st.write("This naive backtest ignores slippage, spreads, fees, latency — real results will be very different.")

# -------------------------------------------------------------------
# Show last few candles + predicted probability / signal
# -------------------------------------------------------------------
st.subheader("Recent candles with model signal (last test window)")
display_n = min(50, len(df_test))
display_df = df_test.copy().tail(display_n)
if use_sklearn and y_prob is not None:
    probs = y_prob[-display_n:]
    display_df = display_df.assign(pred_prob=probs, pred=(probs>0.5).astype(int))
else:
    preds = y_pred[-display_n:]
    display_df = display_df.assign(pred=(preds).astype(int))
st.dataframe(display_df[["time","open","high","low","close","volume","pred_prob" if "pred_prob" in display_df.columns else "pred","pred"]].tail(display_n))

# Price chart with pred overlay
fig = go.Figure(data=[go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])])
# overlay buy signals from test set where pred==1
test_times = df_test["time"].values
test_prices = df_test["close"].values
buy_idxs = np.where(y_pred==1)[0]
buy_times = test_times[buy_idxs]
buy_prices = test_prices[buy_idxs]
fig.add_trace(go.Scatter(x=buy_times, y=buy_prices, mode="markers", marker_symbol="triangle-up", marker_size=8, name="Predicted BUY"))
fig.update_layout(height=600, title="Price with predicted BUY markers (test set)")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# Final candid message and next steps
# -------------------------------------------------------------------
st.markdown("---")
st.markdown("### Final note (please read)")
st.markdown("""
- I cannot deliver 100% predictions. Any model you train must be **backtested** and stress-tested before risking real capital.
- Use realistic metrics (accuracy, precision, recall), and always account for slippage, spread, latency, and fees.
- Good next steps:
  - add more features (orderbook imbalance, trade ticks, VWAP, ATR),
  - use time-series CV (not random splits),
  - add stop-loss / take-profit rules and implement realistic execution simulation,
  - run longer backtests and out-of-sample tests.
""")
