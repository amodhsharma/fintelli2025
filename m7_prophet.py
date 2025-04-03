import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from prophet import Prophet
import streamlit as st

@st.cache_data  
def m7_prophet(data):
    st.markdown("<h3 style='color: cyan;'>M7: Prophet Model</h3>", unsafe_allow_html=True)
    st.write("Prophet is a forecasting tool created by Facebook, designed to handle time series data that may have missing values and seasonal effects. It is particularly effective for daily observations with strong seasonal patterns.")
    
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    
    df_copy['ds'] = df_copy.index  # Use the index (Date) as 'ds'
    df_copy['y'] = df_copy['Close']
    
    train_data = df_copy.sample(frac=0.85, random_state=0)
    test_data = df_copy.drop(train_data.index)
    
    model = Prophet(daily_seasonality=True)
    
    model.fit(train_data)
    
    prediction = model.predict(pd.DataFrame({'ds': test_data.index}))
    
    y_actual = test_data['y']
    y_predicted = prediction['yhat']
    y_predicted = y_predicted.astype(int)
    
    # Plotting with Plotly (only showing past 15% data for the test)
    fig = go.Figure(data=[
        go.Scatter(x=test_data.index, y=y_actual, mode='lines', name='Actual', line=dict(color='blue')),
        go.Scatter(x=test_data.index, y=y_predicted, mode='lines', name='Predicted', line=dict(color='orange'))
    ])
    
    fig.update_layout(
        title="Prophet",
        xaxis_title="Date",
        yaxis_title="Close Price",
        xaxis=dict(rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(showline=True, linecolor="white", linewidth=1),
        template="plotly_dark",
        legend_title="Reference"
    )
    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)
    st.plotly_chart(fig)
    
    blue_text = "color: #3498DB;"
    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)
    st.subheader("Evaluation Metrics")

    mae = mean_absolute_error(y_actual, y_predicted)
    mse = mean_squared_error(y_actual, y_predicted)
    rmse = sqrt(mse)
    r2 = r2_score(y_actual, y_predicted)
    mape = np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100
    
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "R^2": r2,
        "MAPE": mape
    }

    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    predicted_price = y_predicted.iloc[-1] if len(y_predicted) > 0 else None
    st.markdown("`CLOSING PRICE PREDICTION FOR THE DAY`", unsafe_allow_html=True)
    st.metric(label="Prophet", value=f"₹{predicted_price:.2f}" if predicted_price is not None else "N/A")


    # return metrics
