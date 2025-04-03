import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def m8_lstm(data):
    st.markdown("<h1 style='color: cyan;'>Deep Learning for Time Series</h1>", unsafe_allow_html=True),
    st.markdown("<h3 style='color: cyan;'>M8: LSTM Model</h3>", unsafe_allow_html=True),
    st.write("LSTM is a type of Recurrent Neural Network (RNN) designed to capture long-term dependencies in sequential data. It is particularly useful for tasks like time series forecasting, natural language processing, and stock market prediction.")

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
    
    #training model 
    X_train, y_train = [], []
    
    for i in range(1, len(train_data)):
        X_train.append(train_data[i-1:i, 0])
        y_train.append(train_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    #prediction 
    X_test, y_test = [], []
    
    for i in range(1, len(test_data)):
        X_test.append(test_data[i-1:i, 0])
        y_test.append(test_data[i, 0])
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted = model.predict(X_test)
    
    # Inverse transform both predicted and actual values
    predicted = scaler.inverse_transform(predicted)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # FIX: Transform actual values

    
    # Plot the results
    fig = go.Figure(data=[
        go.Scatter(x=df_copy.index[-len(y_test):].to_list(), y=y_test.flatten(), mode='lines', name='Actual', line=dict(color='blue')),
        go.Scatter(x=df_copy.index[-len(y_test):].to_list(), y=predicted.flatten(), mode='lines', name='Predicted', line=dict(color='orange'))
    ])


    fig.update_layout(title='LSTM',
                    xaxis=dict(title="Date",rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
                    yaxis=dict(title=target_column, showline=True, linecolor="white", linewidth=1),
                    legend_title='Reference')
    
    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)
    st.plotly_chart(fig)
    
    # Evaluate the metrics
    mse = mean_squared_error(y_test, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predicted)
    mape = np.mean(np.abs((y_test - predicted) / y_test)) * 100
    r2 = r2_score(y_test, predicted)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R^2': r2
    }
    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)
    st.subheader("Evaluation Metrics")
    blue_text = "color: #3498DB;"
    #st.markdown(f'<p style="{blue_text}"><b>MSE:</b> {metrics["MSE"]:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    predicted_price = predicted[-1, 0] if len(predicted) > 0 else None
    st.markdown("`CLOSING PRICE PREDICTION FOR THE DAY`", unsafe_allow_html=True)
    st.metric(label="LSTM", value=f"₹{predicted_price:.2f}" if predicted_price is not None else "N/A")
