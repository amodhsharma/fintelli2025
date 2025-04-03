import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split

@st.cache_data
def future_xg(data, target_column='Close'):
    st.markdown("<h3 style='color: cyan;'>M6.1: Future Prediction using XG Boost</h3>", unsafe_allow_html=True)

    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['year'] = df_copy.index.year
    df_copy['day'] = df_copy.index.day

    # Define features & target
    X, y = df_copy.drop(columns=[target_column]), df_copy[target_column]
    
    # Train XGBoost model on full data
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    
    # Generate future dates
    last_date = df_copy.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')
    
    # Create future feature set with the same columns as training data
    future_data = pd.DataFrame(index=future_dates)
    future_data['day_of_week'] = future_data.index.dayofweek
    future_data['month'] = future_data.index.month
    future_data['year'] = future_data.index.year
    future_data['day'] = future_data.index.day
    
    # Copy last known values for missing features
    missing_features = [col for col in X.columns if col not in future_data.columns]
    for feature in missing_features:
        future_data[feature] = X[feature].iloc[-1]  # Use last known value
    
    # Predict future prices
    future_predictions = model.predict(future_data[X.columns])  # Ensure correct feature order
    
    y_last_100 = y[-100:]

    fig = go.Figure(data=[
        go.Scatter(x=y_last_100.index, y=y_last_100, mode='lines', name='Last 100 Actual', line=dict(color='blue')),
        go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Predicted Future', line=dict(color='red', dash='dash'))
    ])
    fig.update_layout(
        title='Future Prediction using XG Boost',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        xaxis=dict(rangeslider=dict(visible=True)),
        legend_title='Reference'
    )
    st.markdown("`PRICE OF NEXT 30 DAYS FORCAST`", unsafe_allow_html=True)
    st.plotly_chart(fig)
    