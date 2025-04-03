import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.api import ExponentialSmoothing

@st.cache_data
def future_exponential(data):
    st.markdown("<h3 style='color: cyan;'>M2.1: Future prediction using Exponential Smoothing</h3>", unsafe_allow_html=True)
    
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    target_column = "Close"
    
    train = df_copy.copy()
    
    future_steps = 30
    future_dates = pd.date_range(start=train.index[-1], periods=future_steps + 1, freq='D')[1:]
    
    model = ExponentialSmoothing(train['Close'], seasonal='add', seasonal_periods=12, trend='add').fit()
    forecast_values = model.forecast(future_steps)
    
    last_100_train = train.tail(100)  # Last 100 entries of training data
    
    fig = go.Figure(data=[
        go.Scatter(x=last_100_train.index, y=last_100_train['Close'], mode='lines', name='Last 100 Train Data', line=dict(color='blue')),
        go.Scatter(x=future_dates, y=forecast_values, mode='lines', name='Predicted Next 30 Days', line=dict(color='red', dash='dot'))
    ])
    
    fig.update_layout(
        title="Future prediction using Exponential Smoothing",
        xaxis_title='Date',
        yaxis_title='Stock Price',
        xaxis=dict(rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(showline=True, linecolor="white", linewidth=1),
        legend_title='Reference',
    )
    
    st.markdown("`PRICE OF NEXT 30 DAYS FORCAST`", unsafe_allow_html=True)
    st.plotly_chart(fig)
    
