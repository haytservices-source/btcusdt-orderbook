import requests
import pandas as pd

SYMBOL = "BTCUSDT"

# --- Get latest price ---
price_data = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={SYMBOL}").json()
price = float(price_data["price"])
print(f"BTC/USDT Futures Price: ${price:,.2f}")

# --- Get order book (Top 5) ---
order_book = requests.get(f"https://fapi.binance.com/fapi/v1/depth?symbol={SYMBOL}&limit=5").json()

bids = order_book["bids"]
asks = order_book["asks"]

buyers_strength = sum(float(bid[1]) for bid in bids)
sellers_strength = sum(float(ask[1]) for ask in asks)
total_volume = buyers_strength + sellers_strength

df_bids = pd.DataFrame(bids, columns=["Bid Price", "Bid Qty"])
df_asks = pd.DataFrame(asks, columns=["Ask Price", "Ask Qty"])
df_orderbook = pd.concat([df_bids, df_asks], axis=1)

print("\nTop 5 Order Book Levels:")
print(df_orderbook)
print(f"\nBuyers Strength: {buyers_strength:.4f} BTC")
print(f"Sellers Strength: {sellers_strength:.4f} BTC")
print(f"Total Volume (Top 5): {total_volume:.4f} BTC")
