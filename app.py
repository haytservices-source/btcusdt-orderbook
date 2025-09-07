import streamlit as st
import requests

st.title("BTC/USDT (USDⓈ-M Futures) Live Price")

SYMBOL = "BTCUSDT"
PRICE_URL = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={SYMBOL}"

try:
    response = requests.get(PRICE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=2)
    response.raise_for_status()
    price = float(response.json()['price'])
except Exception as e:
    price = 0.0
    st.error(f"Failed to fetch price: {e}")

st.metric(label="BTC/USDT Price", value=f"${price:,.2f}")
