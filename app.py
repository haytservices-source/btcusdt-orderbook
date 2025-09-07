import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="BTCUSDT Order Book", layout="wide")
SYMBOL = "BTCUSDT"
LIMIT = 20
BASE_URL = "https://api.binance.us/api/v3"

# Function to fetch order book
def get_orderbook():
    url = f"{BASE_URL}/depth?symbol={SYMBOL}&limit={LIMIT}"
    response = requests.get(url)
    data = response.json()

    # Check if bids/asks exist
    if 'bids' in data and 'asks' in data:
        bids = pd.DataFrame(data['bids'], columns=['Price','Quantity']).astype(float)
        asks = pd.DataFrame(data['asks'], columns=['Price','Quantity']).astype(float)
        return bids, asks
    else:
        raise ValueError(f"Order book data not available: {data}")

st.title("BTC/USDT Live Order Book (Binance US)")

# Placeholder for updating the chart
placeholder = st.empty()

while True:
    try:
        bids, asks = get_orderbook()
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bids['Price'], y=bids['Quantity'], name='Bids', marker_color='green'))
        fig.add_trace(go.Bar(x=asks['Price'], y=asks['Quantity'], name='Asks', marker_color='red'))
        fig.update_layout(
            title=f"{SYMBOL} Order Book (Top {LIMIT})",
            barmode='overlay',
            xaxis_title='Price',
            yaxis_title='Quantity',
            xaxis=dict(type='category'),
        )
        
        # Update chart in place with a unique key
        placeholder.plotly_chart(fig, use_container_width=True, key="orderbook_chart")

        time.sleep(1)
    
    except Exception as e:
        st.error(f"Error fetching order book: {e}")
        time.sleep(5)
