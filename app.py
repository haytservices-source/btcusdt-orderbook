import streamlit as st
import requests
import time

st.set_page_config(page_title="BTC/USDT Live Price", layout="wide")
st.title("BTC/USDT Live Price (Binance US)")

price_placeholder = st.empty()

def get_price():
    try:
        url = "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except Exception:
        return None

while True:
    price = get_price()
    if price:
        price_placeholder.metric(label="BTC/USDT Price", value=f"${price:,.2f}")
    else:
        price_placeholder.text("❌ Failed to fetch price from Binance US API.")
    time.sleep(1)
