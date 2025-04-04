import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import streamlit as st

@st.cache_data
def extraction_m4(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    # 85-15 train-test split
    train_size = int(len(df_copy) * 0.85)
    train, test = df_copy.iloc[:train_size], df_copy.iloc[train_size:]

    # Train SARIMA model
    model = SARIMAX(train['Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=len(test)).predicted_mean

    # Compute RMSE
    def evaluate_forecast(actual, predicted):
        min_len = min(len(actual), len(predicted))
        actual, predicted = actual[:min_len], predicted[:min_len]
        return np.sqrt(mean_squared_error(actual, predicted))

    rmse = evaluate_forecast(test['Close'].values, forecast.values)

    # Get last actual and predicted price
    last_actual_price = test['Close'].iloc[-1]
    last_predicted_price = forecast.iloc[-1] if not forecast.empty else None

    return rmse, last_actual_price, last_predicted_price
