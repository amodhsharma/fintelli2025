import streamlit as st
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

@st.cache_data
def future_sarima(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    st.markdown("<h3 style='color: cyan;'>M4.1: Future Prediction using SARIMA</h3>", unsafe_allow_html=True)

    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    
    # Train model on full dataset
    model = SARIMAX(df_copy['Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    # Generate future dates for the next month
    last_date = df_copy.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')  # 'B' ensures business days

    # Forecast next 30 days
    forecast = model_fit.get_forecast(steps=30).predicted_mean

    # Prepare data for plotting
    historical_data = df_copy['Close'].iloc[-100:]  # Last 100 actual stock prices

    fig = go.Figure(data=[
        go.Scatter(x=historical_data.index, y=historical_data, mode='lines', name='Historical Data', line=dict(color='blue')),
        go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast (Next 1 Month)', line=dict(color='red', dash='dot'))
    ])

    fig.update_layout(
        title="Future Prediction using SARIMA",
        xaxis=dict(title="Date", rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(title="Stock Price", showline=True, linecolor="white", linewidth=1),
        legend_title='Reference'
    )
    st.markdown("`PRICE OF NEXT 30 DAYS FORCAST`", unsafe_allow_html=True)
    st.plotly_chart(fig)
