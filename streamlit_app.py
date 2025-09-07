import streamlit as st
import threading
import time
import random

# ---- Simulated predictor (replace with your Binance/WebSocket logic) ----
latest = {"price": None, "prediction": "Starting...", "score": 0.0}

def fake_predictor():
    while True:
        price = random.uniform(60000, 65000)  # simulate BTC price
        direction = random.choice(["UP 📈", "DOWN 📉"])
        score = random.random()
        latest.update({"price": round(price, 2), "prediction": direction, "score": score})
        time.sleep(2)

# Start background thread only once
if "thread_started" not in st.session_state:
    t = threading.Thread(target=fake_predictor, daemon=True)
    t.start()
    st.session_state.thread_started = True

# ---- Streamlit UI ----
st.set_page_config(page_title="BTCUSDT Predictor", layout="wide")
st.title("BTC/USDT Live Predictor")

placeholder = st.empty()

while True:
    price = latest["price"]
    prediction = latest["prediction"]
    score = latest["score"]
    placeholder.markdown(f"**Price:** {price}  \n**Prediction:** {prediction}  \n**Confidence:** {score:.2f}")
    time.sleep(1)
