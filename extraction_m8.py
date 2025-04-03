import numpy as np
import pandas as pd
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st

@st.cache_data
def extraction_m8(data):
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    target_column = 'Close'  
    data = df_copy[[target_column]].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split the data into training (85%) and testing (15%)
    train_size = int(len(scaled_data) * 0.85)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Create and train the LSTM model
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

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Prepare test data
    X_test, y_test = [], []
    for i in range(1, len(test_data)):
        X_test.append(test_data[i-1:i, 0])
        y_test.append(test_data[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make predictions
    predicted = model.predict(X_test)

    # Inverse transform both predicted and actual values
    predicted = scaler.inverse_transform(predicted)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Compute RMSE
    rmse = sqrt(mean_squared_error(y_test, predicted))

    # Get last actual and predicted closing price
    last_actual_price = y_test[-1, 0]
    last_predicted_price = predicted[-1, 0] if len(predicted) > 0 else None

    return rmse, last_actual_price, last_predicted_price
