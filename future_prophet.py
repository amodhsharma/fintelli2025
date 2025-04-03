import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import streamlit as st

@st.cache_data
def future_prophet(data):
    st.markdown("<h3 style='color: cyan;'>M7.1:Future prediction using Prophet</h3>", unsafe_allow_html=True)

    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    
    # Prepare data for Prophet
    df_copy['ds'] = df_copy.index  # Date as 'ds'
    df_copy['y'] = df_copy['Close']  # Closing price as 'y'
    
    # Train the model on the full dataset
    model = Prophet(daily_seasonality=True)
    model.fit(df_copy)
    
    # Create future dataframe for the next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Extract relevant parts of the forecast
    future_dates = forecast['ds'][-30:]
    predicted_prices = forecast['yhat'][-30:]
    
    # Plot actual vs predicted
    fig = go.Figure(data=[
        go.Scatter(x=df_copy['ds'][-100:], y=df_copy['y'][-100:], mode='lines', name='Actual (Last 100)', line=dict(color='blue')),
        go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name='Forecast (Next 30 Days)', line=dict(color='orange', dash='dot'))
    ])

    # Update layout with range slider
    fig.update_layout(
        title="Future prediction using Prophet",
        xaxis_title="Date",
        yaxis_title="Close Price",
        xaxis=dict(rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(showline=True, linecolor="white", linewidth=1),
        template="plotly_dark",
        legend_title="Legend"
    )
    st.markdown("`PRICE OF NEXT 30 DAYS FORCAST`", unsafe_allow_html=True)
    st.plotly_chart(fig)
