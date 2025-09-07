import streamlit as st
import requests
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="💎 Golden Sniper Pro BTC/USDT Predictor", layout="wide")

# Auto-refresh every 1 second
st_autorefresh(interval=1000, key="refresh")

# Function to get latest price from Binance US
def get_latest_price():
    url = "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT"
    response = requests.get(url)
    data = response.json()
    return float(data["price"])

# Get latest price
price = get_latest_price()

# Simple trend logic
trend = "WAIT"
confidence = 50
if "prev_price" not in st.session_state:
    st.session_state.prev_price = price

if price > st.session_state.prev_price * 1.0005:
    trend = "UP"
    confidence = 80
elif price < st.session_state.prev_price * 0.9995:
    trend = "DOWN"
    confidence = 80

st.session_state.prev_price = price

# Display
st.markdown(f"**Price:** ${price:,.2f}")
st.markdown(f"**Trend Signal:** {trend}")
st.markdown(f"**Confidence:** {confidence}%")
