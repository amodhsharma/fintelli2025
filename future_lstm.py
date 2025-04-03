import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def future_lstm(data):
    st.markdown("<h3 style='color: cyan;'>M8.1: Future prediction using LSTM</h3>", unsafe_allow_html=True)

    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    target_column = 'Close'

    # Extract and scale data
    data_values = df_copy[[target_column]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(data_values)

    # Prepare LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(1, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Prepare training data
    X_train, y_train = [], []
    for i in range(1, len(train_data)):
        X_train.append(train_data[i-1:i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predict next 30 days
    future_days = 30
    last_100 = train_data[-100:].reshape(-1, 1)
    future_predictions = []
    input_seq = last_100[-1]

    for _ in range(future_days):
        input_seq = np.reshape(input_seq, (1, 1, 1))
        pred = model.predict(input_seq, verbose=0)
        future_predictions.append(pred[0, 0])
        input_seq = np.array([[pred[0, 0]]])

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Get actual last 100 days data
    last_100_actual = scaler.inverse_transform(train_data[-100:].reshape(-1, 1))
    last_100_dates = df_copy.index[-100:]

    # Generate future dates
    future_dates = [last_100_dates[-1] + timedelta(days=i) for i in range(1, future_days + 1)]

    # Plot results
    fig = go.Figure(data=[
        go.Scatter(x=last_100_dates, y=last_100_actual.flatten(), mode='lines', name='Actual (Last 100)', line=dict(color='blue')),
        go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name='Predicted (Next 30 Days)', line=dict(color='orange', dash='dot'))
    ])
    fig.update_layout(
        title='Future Prediction using LSTM',
        xaxis=dict(title='Date', rangeslider=dict(visible=True), showline=True, linecolor='white', linewidth=1),
        yaxis=dict(title='Closing Price', showline=True, linecolor='white', linewidth=1),
        legend_title='Reference'
    )

    st.markdown("`PRICE OF NEXT 30 DAYS FORECAST`", unsafe_allow_html=True)
    st.plotly_chart(fig)
