import streamlit as st
import time
import random

st.set_page_config(page_title="BTCUSDT Predictor", layout="wide")
st.title("BTC/USDT Live Predictor")

# --- Fake predictor (replace later with Binance logic) ---
def get_prediction():
    price = random.uniform(60000, 65000)
    direction = random.choice(["UP 📈", "DOWN 📉"])
    score = random.random()
    return {"price": round(price, 2), "prediction": direction, "score": score}

# --- Auto-refresh ---
# This forces Streamlit to rerun the script every 5 seconds
st_autorefresh = st.experimental_data_editor if False else None  # keep linter happy
st_autorefresh = st.experimental_rerun if False else None        # placeholder

st_autorefresh = st.experimental_rerun  # but don't call directly

# Instead of rerun, use st_autorefresh widget
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=5000, key="btc_refresh")  # 5000 ms = 5 sec refresh

# --- Show prediction ---
data = get_prediction()
st.metric("Price", data["price"])
st.metric("Prediction", data["prediction"])
st.metric("Confidence", f"{data['score']:.2f}")
