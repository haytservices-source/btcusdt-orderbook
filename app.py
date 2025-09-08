from binance.um_futures import UMFutures
import pandas as pd

# Initialize Binance Futures client (no API key needed for public market data)
client = UMFutures()

SYMBOL = "BTCUSDT"

# --- Get latest price ---
price_data = client.ticker_price(SYMBOL)
price = float(price_data["price"])
print(f"BTC/USDT Futures Price: ${price:,.2f}")

# --- Get order book (Top 5) ---
order_book = client.depth(SYMBOL, limit=5)

bids = order_book["bids"]
asks = order_book["asks"]

# Calculate buyer/seller strength
buyers_strength = sum(float(bid[1]) for bid in bids)
sellers_strength = sum(float(ask[1]) for ask in asks)
total_volume = buyers_strength + sellers_strength

# Format as DataFrame
df_bids = pd.DataFrame(bids, columns=["Bid Price", "Bid Qty"])
df_asks = pd.DataFrame(asks, columns=["Ask Price", "Ask Qty"])
df_orderbook = pd.concat([df_bids, df_asks], axis=1)

# --- Print results ---
print("\nTop 5 Order Book Levels:")
print(df_orderbook)
print(f"\nBuyers Strength: {buyers_strength:.4f} BTC")
print(f"Sellers Strength: {sellers_strength:.4f} BTC")
print(f"Total Volume (Top 5): {total_volume:.4f} BTC")
