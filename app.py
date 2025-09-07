import streamlit as st
import random
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="BTCUSDT Predictor", layout="wide")
st.title("BTC/USDT Live Predictor")

# --- Auto-refresh every 5 seconds ---
st_autorefresh(interval=5000, key="btc_refresh")

# --- Fake predictor (replace later with Binance logic) ---
def get_prediction():
    price = random.uniform(60000, 65000)
    direction = random.choice(["UP 📈", "DOWN 📉"])
    score = random.random()
    return {"price": round(price, 2), "prediction": direction, "score": score}

# --- Show prediction ---
data = get_prediction()
st.metric("Price", data["price"])
st.metric("Prediction", data["prediction"])
st.metric("Confidence", f"{data['score']:.2f}")
