import pandas as pd
import streamlit as st
import plotly.graph_objects as go

@st.cache_data
def eda_volatility(data):
    st.markdown("<h3 style='color: cyan;'>EDA: Stock Price Volatility</h3>", unsafe_allow_html=True)
    st.write("The degree of variation in a stock's price over time, often measured by standard deviation or the average absolute change in price. 30-day volatility tracks price fluctuations over the past month, while 90-day volatility captures price movements over a three-month period, helping you assess short-term versus medium-term risk.")
    
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    df_copy['Volatility_30'] = df_copy['Close'].rolling(window=30).std()
    df_copy['Volatility_90'] = df_copy['Close'].rolling(window=90).std()
    
    # Create figure
    fig = go.Figure(data=[
        go.Scatter(x=df_copy.index, y=df_copy['Volatility_30'], mode='lines', name='30-Day Rolling Volatility', line=dict(color='blue')),
        go.Scatter(x=df_copy.index, y=df_copy['Volatility_90'], mode='lines', name='90-Day Rolling Volatility', line=dict(color='red'))
    ])
    fig.update_layout(
        autosize=True,
        title="Stock Price Volatility",
        legend_title='Reference',
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),  # Enable range slider
            type="date",
            showline=True,
            linecolor="white",  # Black x-axis line
            linewidth=1  # Make it bold
        ),
        yaxis=dict(title="Volatility",
            showline=True,  # Show y-axis line
            linecolor="white",  # Black y-axis line
            linewidth=1  # Make it bold
        )
    )
    # Configure layout
    st.markdown("`EXPLORATORY DATA ANALYSIS`", unsafe_allow_html=True)
    st.plotly_chart(fig)
