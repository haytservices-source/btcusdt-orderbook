import streamlit as st
import requests
import time

st.title("BTC/USDT (USDⓈ-M Futures) Live Price")

price_placeholder = st.empty()

# Function to fetch BTCUSDT price from Binance Futures API
def get_price():
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"
        response = requests.get(url, timeout=5)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        return None

# Auto-refresh every 0.5 second
while True:
    price = get_price()
    if price:
        price_placeholder.metric(label="BTC/USDT Price", value=f"${price:,.2f}")
    else:
        price_placeholder.text("Failed to fetch price.")
    time.sleep(0.5)
