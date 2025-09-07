import streamlit as st
import pandas as pd
import asyncio
import websockets
import json

st.set_page_config(page_title="BTC/USDT (USDⓈ-M Futures) Live Order Book Dashboard", layout="wide")
st.title("BTC/USDT (USDⓈ-M Futures) Live Order Book Dashboard")

# Placeholders for dynamic updates
price_placeholder = st.empty()
buyers_placeholder = st.empty()
sellers_placeholder = st.empty()
volume_placeholder = st.empty()
orderbook_placeholder = st.empty()

# Binance Futures WebSocket URL for top 5 depth updates (100ms)
WS_URL = "wss://fstream.binance.com/ws/btcusdt@depth5@100ms"

async def main():
    async with websockets.connect(WS_URL) as ws:
        while True:
            try:
                data = await ws.recv()
                data_json = json.loads(data)

                # Extract bids and asks
                bids = data_json.get("b", [])  # list of [price, qty]
                asks = data_json.get("a", [])

                # Calculate Buyers/Sellers Strength
                buyers_strength = sum([float(bid[1]) for bid in bids])
                sellers_strength = sum([float(ask[1]) for ask in asks])
                total_volume = buyers_strength + sellers_strength

                # Current price: mid of top bid and ask
                price = (float(bids[0][0]) + float(asks[0][0])) / 2 if bids and asks else 0.0

                # Prepare order book DataFrame
                df_bids = pd.DataFrame(bids, columns=['Bid Price', 'Bid Qty'])
                df_asks = pd.DataFrame(asks, columns=['Ask Price', 'Ask Qty'])
                df_orderbook = pd.concat([df_bids, df_asks], axis=1)

                # Update Streamlit placeholders
                price_placeholder.markdown(f"**Price:** ${price:,.2f}")
                buyers_placeholder.markdown(f"**Buyers Strength:** {buyers_strength:.4f} BTC")
                sellers_placeholder.markdown(f"**Sellers Strength:** {sellers_strength:.4f} BTC")
                volume_placeholder.markdown(f"**Total Volume (Top 5):** {total_volume:.4f} BTC")
                orderbook_placeholder.table(df_orderbook)

            except Exception as e:
                st.error(f"Error: {e}")

# Run the WebSocket loop in Streamlit
asyncio.run(main())
