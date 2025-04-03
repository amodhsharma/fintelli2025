import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import timedelta

@st.cache_data
def future_linear(data):
    st.markdown("<h3 style='color: cyan;'>M1.2: Future Predection using Linear Regression Model</h3>", unsafe_allow_html=True)
    
    # Preprocess data
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    target_column = "Close"

    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]
    
    # Train Model on 100% of Available Data
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict Next 30 Days
    last_date = X.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    future_X = X.iloc[-1:].values  # Last available row
    future_predictions = []
    
    for _ in range(30):
        pred = model.predict(future_X)[0]
        future_predictions.append(pred)
        future_X = np.roll(future_X, -1)
        future_X[0, -1] = pred  # Updating with the last predicted value
    
    # Select last 300 entries from training data
    recent_X = df_copy.index[-100:] if len(df_copy) > 100 else df_copy.index
    recent_y = y[-100:] if len(y) > 100 else y
    
    # Plot Actual vs Predicted Prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_X, y=recent_y, mode='lines', name='Recent Training Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Forecast (Next 30 Days)', line=dict(color='red', dash='dash')))
    
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Closing Price',
        legend_title='Legend',
        xaxis=dict(rangeslider=dict(visible=True))
    )
    st.markdown("`PRICE OF NEXT 30 DAYS FORCAST`", unsafe_allow_html=True)
    st.plotly_chart(fig)