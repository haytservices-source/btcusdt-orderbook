import streamlit as st
import requests

st.set_page_config(page_title="BTC/USDT Live Price", layout="wide")
st.title("BTC/USDT Live Price (Binance)")

price_placeholder = st.empty()

def get_price():
    urls = {
        "Futures": "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT",
        "Spot": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        "US": "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT"
    }
    for source, url in urls.items():
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            if "price" in data:   # ✅ avoid KeyError
                return float(data["price"]), source
        except:
            continue
    return None, None

price, source = get_price()

if price:
    price_placeholder.metric(
        label=f"BTC/USDT Price ({source})",
        value=f"${price:,.2f}"
    )
else:
    price_placeholder.error("❌ Could not fetch price from Binance APIs.")
