# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

@st.cache_data
def m4_sarima(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    st.markdown("<h3 style='color: cyan;'>M4: SARIMA - Seasonal Autoregressive Integrated Moving Average</h3>", unsafe_allow_html=True)
    st.write("SARIMA is an extension of ARIMA that supports seasonal differencing. It is particularly useful for time series data with seasonal patterns.")
    
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    # Ensure stationarity with differencing if necessary
    if df_copy['Close'].diff().dropna().var() > df_copy['Close'].var() * 0.01:
        df_copy['Close'] = df_copy['Close'].diff().dropna()
    
    # 85-15 train-test split
    train_size = int(len(df_copy) * 0.85)
    train, test = df_copy.iloc[:train_size], df_copy.iloc[train_size:]
    
    model = SARIMAX(train['Close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=len(test)).predicted_mean
    
    fig = go.Figure(data=[
        go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train Data', line=dict(color='blue')),
        go.Scatter(x=test.index, y=test['Close'], mode='lines', name='Test Data', line=dict(color='green')),
        go.Scatter(x=test.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red', dash='dot'))
    ])
    fig.update_layout(title="SARIMA",
        xaxis=dict(title="Date", rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(title="Stock Price", showline=True, linecolor="white", linewidth=1),
        legend_title="Reference",
    )
    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)
    st.plotly_chart(fig)
    
    # Call the evaluation function
    #evaluate_forecast(test['Close'].values, forecast.values)
    
    #evaluation
    @st.cache_data
    def evaluate_forecast(actual, predicted):
        min_len = min(len(actual), len(predicted))
        actual, predicted = actual[:min_len], predicted[:min_len]
        # actual = actual[:min_len]  
        # predicted = predicted[:min_len]  

        if len(actual) == 0 or len(predicted) == 0:  # Prevent NoneType errors
            return{"RMSE": None, "MAE": None, "MAPE": None, "R^2": None}

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted) * 100  # in percentage
        r2 = r2_score(actual, predicted)

        return{
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R^2": r2
        }

    # metrics = evaluate_forecast(test['Close'].values, forecast.values)
    metrics = evaluate_forecast(test['Close'].values, forecast.values if forecast is not None else [])

    if forecast is not None and len(forecast) > 0:
        predicted_price = forecast.values[-1]
    else:
        predicted_price = None

    blue_text = "color: #3498DB;"

    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)
    st.subheader("Evaluation Metrics")
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    predicted_price = forecast.values[-1] if len(forecast.values) > 0 else None
    st.markdown("`CLOSING PRICE PREDECTION FOR THE DAY`", unsafe_allow_html=True)
    st.metric(label="SARIMA", value=f"₹{predicted_price:.2f}" if predicted_price else "N/A")