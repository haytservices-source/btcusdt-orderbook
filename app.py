import requests
import json

# === CONFIG ===
OANDA_ACCOUNT_ID = "YOUR_ACCOUNT_ID"
OANDA_API_KEY = "YOUR_API_KEY"
OANDA_URL = "https://api-fxpractice.oanda.com/v3"

headers = {
    "Authorization": f"Bearer {OANDA_API_KEY}",
    "Content-Type": "application/json"
}

# === Live Price ===
def get_live_price():
    url = f"{OANDA_URL}/accounts/{OANDA_ACCOUNT_ID}/pricing?instruments=XAU_USD"
    response = requests.get(url, headers=headers)
    data = response.json()
    price = data['prices'][0]
    print(f"XAU/USD Live Price: Bid: {price['bids'][0]['price']} / Ask: {price['asks'][0]['price']}")

# === Order Book ===
def get_order_book():
    url = f"{OANDA_URL}/instruments/XAU_USD/orderBook"
    response = requests.get(url, headers=headers)
    data = response.json()
    print("\n--- XAU/USD Order Book ---")
    for bucket in data['orderBook']['buckets'][:10]:  # top 10 buckets
        print(f"Price: {bucket['price']}, Long %: {bucket['longCountPercent']}, Short %: {bucket['shortCountPercent']}")

if __name__ == "__main__":
    get_live_price()
    get_order_book()
