import streamlit as st
import time
import random

st.set_page_config(page_title="BTCUSDT Predictor", layout="wide")
st.title("BTC/USDT Live Predictor")

# Fake data (replace with Binance logic later)
def get_prediction():
    price = random.uniform(60000, 65000)
    direction = random.choice(["UP 📈", "DOWN 📉"])
    score = random.random()
    return {"price": round(price, 2), "prediction": direction, "score": score}

# Auto-refresh every 2 seconds
count = st.experimental_rerun if False else None  # placeholder for rerun
placeholder = st.empty()

for i in range(1000):  # run limited times
    data = get_prediction()
    with placeholder.container():
        st.markdown(f"""
        **Price:** {data['price']}  
        **Prediction:** {data['prediction']}  
        **Confidence:** {data['score']:.2f}
        """)
    time.sleep(2)
    st.experimental_rerun()  # refresh UI
