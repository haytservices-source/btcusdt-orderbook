import requests
import time
import os

# ✅ Use Futures endpoint instead of Spot (works even if spot is blocked in your region)
BASE_URL = "https://fapi.binance.com/fapi/v1/depth"

SYMBOL = "BTCUSDT"
LIMIT = 20  # number of bids/asks
REFRESH = 1  # refresh every 1 second

def fetch_orderbook():
    try:
        url = f"{BASE_URL}?symbol={SYMBOL}&limit={LIMIT}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to load order book data: {e}")
        return None

while True:
    os.system("clear")  # clear screen (use "cls" if on Windows)
    data = fetch_orderbook()
    
    if data:
        bids = data['bids']
        asks = data['asks']

        print(f"\n📊 Order Book for {SYMBOL} (Top {LIMIT})\n")
        print("---- BUYERS (Bids) ----")
        for price, qty in bids[:5]:
            print(f"Buy {qty} @ {price}")

        print("\n---- SELLERS (Asks) ----")
        for price, qty in asks[:5]:
            print(f"Sell {qty} @ {price}")

        # Quick imbalance check
        total_bid = sum(float(q) for _, q in bids)
        total_ask = sum(float(q) for _, q in asks)

        print("\n📈 Analysis:")
        if total_bid > total_ask:
            print("🔥 Buyers stronger → Possible UP move")
        elif total_ask > total_bid:
            print("⚠️ Sellers stronger → Possible DOWN move")
        else:
            print("➖ Balanced order book → Sideways")
    
    time.sleep(REFRESH)
