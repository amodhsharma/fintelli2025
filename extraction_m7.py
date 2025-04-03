import numpy as np
import pandas as pd
from math import sqrt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import streamlit as st

@st.cache_data
def extraction_m7(data):
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    df_copy['ds'] = df_copy.index  # Use the index (Date) as 'ds'
    df_copy['y'] = df_copy['Close']
    
    # Split into training (85%) and test (15%)
    train_data = df_copy.sample(frac=0.85, random_state=0)
    test_data = df_copy.drop(train_data.index)
    
    model = Prophet(daily_seasonality=True)
    model.fit(train_data)
    
    # Predict on test data
    prediction = model.predict(pd.DataFrame({'ds': test_data.index}))
    
    y_actual = test_data['y']
    y_predicted = prediction['yhat'].astype(int)  # Convert to integer values
    
    # Compute RMSE
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    
    # Get last actual and predicted closing price
    last_actual_price = y_actual.iloc[-1]
    last_predicted_price = y_predicted.iloc[-1] if len(y_predicted) > 0 else None

    return rmse, last_actual_price, last_predicted_price
