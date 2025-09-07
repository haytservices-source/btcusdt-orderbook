import streamlit as st
import requests
import time

st.set_page_config(page_title="💎 Golden Sniper Pro BTC/USDT Predictor", layout="wide")

# Create placeholders
price_placeholder = st.empty()
signal_placeholder = st.empty()
confidence_placeholder = st.empty()

# Function to fetch BTC/USDT price from Binance.US
def fetch_binance_price():
    url = "https://api.binance.us/api/v3/ticker/24hr?symbol=BTCUSDT"
    data = requests.get(url).json()
    return float(data['lastPrice'])

# Dummy trend logic (replace with your real logic)
def get_trend_signal(price):
    if price % 2 > 1:  # dummy condition
        return "UP", 0.8
    else:
        return "DOWN", 0.5

# Live updating loop
while True:
    try:
        price = fetch_binance_price()
        trend, confidence = get_trend_signal(price)
        
        # Update placeholders
        price_placeholder.markdown(f"**Price:** ${price:,.2f}")
        signal_placeholder.markdown(f"**Trend Signal:** {trend}")
        confidence_placeholder.markdown(f"**Confidence:** {int(confidence*100)}%")
        
        time.sleep(1)  # update every second
    except Exception as e:
        st.error(f"Error: {e}")
        time.sleep(5)
