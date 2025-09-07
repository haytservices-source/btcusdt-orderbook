import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="BTCUSDT Order Book", layout="wide")
SYMBOL = "BTCUSDT"
LIMIT = 20  # top 20 bids/asks

# Function to fetch order book from Binance REST API
def get_orderbook():
    url = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit={LIMIT}"
    data = requests.get(url).json()
    bids = pd.DataFrame(data['bids'], columns=['Price','Quantity']).astype(float)
    asks = pd.DataFrame(data['asks'], columns=['Price','Quantity']).astype(float)
    return bids, asks

st.title("BTC/USDT Live Order Book")

# Placeholder for live chart
placeholder = st.empty()

# Continuous loop to refresh every second
while True:
    try:
        bids, asks = get_orderbook()
        
        # Plot using Plotly
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
        
        # Render chart in Streamlit
        placeholder.plotly_chart(fig, use_container_width=True)
        
        # Refresh every second
        time.sleep(1)
    
    except Exception as e:
        st.error(f"Error fetching order book: {e}")
        time.sleep(5)

