import streamlit as st
import requests
import time

st.set_page_config(page_title="BTC/USDT Live Price", layout="wide")
st.title("BTC/USDT (USDⓈ-M Futures) Live Price")

price_placeholder = st.empty()

def get_price():
    urls = [
        "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT",  # Futures
        "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"     # Spot fallback
    ]
    headers = {"User-Agent": "Mozilla/5.0"}  # helps avoid blocking

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=3)
            response.raise_for_status()
            data = response.json()
            return float(data['price']), url
        except Exception:
            continue

    return None, None

# Auto-refresh loop
while True:
    price, source = get_price()
    if price:
        price_placeholder.metric(
            label=f"BTC/USDT Price ({'Futures' if 'fapi' in source else 'Spot'})",
            value=f"${price:,.2f}"
        )
    else:
        price_placeholder.text("❌ Failed to fetch price from Binance APIs.")

    time.sleep(1)  # update every second
