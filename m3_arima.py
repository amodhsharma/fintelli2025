#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

@st.cache_data
def m3_arima(data, order=(6,1,0)):
    st.markdown("<h3 style='color: cyan;'>M3: ARIMA - Autoregressive Integrated Moving Average", unsafe_allow_html=True),
    st.markdown("""
    ARIMA is a widely used statistical method for **time series forecasting**, combining three components:  

    - **AR (Autoregression)** → Uses past values to predict future values.  
    - **I (Integration/Differencing)** → Makes the series stationary by removing trends.  
    - **MA (Moving Average)** → Models the relationship between an observation and residual errors.  

    ARIMA is defined by three parameters:  

    - **p** → Number of lag observations (lag order).  
    - **d** → Number of times the data is differenced to remove trends.  
    - **q** → Size of the moving average window.  
    """)

    #st.write("Decomposition plots for Arima")
    #decomposition = seasonal_decompose(train['Close'], model='additive', period=30)
    # fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    # decomposition.trend.plot(ax=axes[0], title='Trend')
    # decomposition.seasonal.plot(ax=axes[1], title='Seasonality')
    # decomposition.resid.plot(ax=axes[2], title='Residuals')
    # st.pyplot(fig)
    
    # st.write("ACF and PACF plots for Arima")
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # plot_acf(train['Close'], ax=axes[0])
    # plot_pacf(train['Close'], ax=axes[1])
    # st.pyplot(fig)

    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    train_size = int(len(df_copy) * 0.85)
    train, test = df_copy[:train_size], df_copy[train_size:]
    
    model = ARIMA(train['Close'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    
    fig = go.Figure(data=[
        go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train', line=dict(color='blue')),
        go.Scatter(x=test.index, y=test['Close'], mode='lines', name='Test', line=dict(color='green')),
        go.Scatter(x=test.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red', dash='dot'))
    ])

    fig.update_layout(title="ARIMA", xaxis_title='Date',
        xaxis=dict(title="Date",rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(title='Stock Price', showline=True, linecolor="white", linewidth=1),
        legend_title='Reference',
    )
    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)
    st.plotly_chart(fig)
    
    #forcast the prices
    @st.cache_data
    def evaluate_forecast(actual, predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]  
        predicted = predicted[:min_len]  

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
    
    metrics = evaluate_forecast(test['Close'].values, forecast.values)

    blue_text = "color: #3498DB;"

    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)

    st.subheader("Evaluation Metrics")
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    predicted_price = forecast.iloc[-1] if not forecast.empty else None
    st.markdown("`CLOSING PRICE PREDECTION FOR THE DAY`", unsafe_allow_html=True)
    st.metric(label="ARIMA", value=f"₹{predicted_price:.2f}" if predicted_price else "N/A")
