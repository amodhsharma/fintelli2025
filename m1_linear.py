import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@st.cache_data
def process_and_train_model(data):
    st.markdown("<h1 style='color: cyan;'>Traditional Regression & Statistical Models</h1>", unsafe_allow_html=True),
    st.markdown("<h3 style='color: cyan;'>M1: Linear Regression Model</h3>", unsafe_allow_html=True),
    st.write("Linear Regression is a fundamental supervised machine learning algorithm used for modeling the relationship between a dependent variable (target) and one or more independent variables (predictors).")
    st.latex(r"y = mx + c")
    st.write("Y → Dependent variable (Target) → The value we want to predict (e.g., future stock price).")
    st.write("X → Independent variable (Predictor) → The input used to predict Y (e.g., time, past stock prices).")
    st.write("m → Slope (Coefficient) → Represents how much Y changes when X increases by 1 unit.")
    st.write("c → Y-intercept → The value of Y when X is zero.")

    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    target_column = "Close"
    
    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R^2': r2}
    
    fig = go.Figure(data=[
        go.Scatter(x=X_test.index, y=y_test, mode='lines', name='Actual', line=dict(color='blue')),
        go.Scatter(x=X_test.index, y=y_pred, mode='lines', name='Predicted', line=dict(color='orange'))
    ])
    fig.update_layout(
        title='Linear Regression', 
        legend_title='Reference', 
        xaxis=dict(title="Date",rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1), 
        yaxis=dict(title=target_column, showline=True, linecolor="white", linewidth=1)
    )
    st.plotly_chart(fig)
    
    
    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)
    st.subheader("Evaluation Metrics")

    blue_text = "color: #3498DB;" 
    #st.markdown(f'<p style="{blue_text}"><b>MSE:</b> {metrics["MSE"]:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    predicted_price = y_pred[-1]
    last_actual_price = y_test.iloc[-1]
    
    st.markdown("`CLOSING PRICE PREDICTION FOR THE DAY`", unsafe_allow_html=True)
    st.metric(label="Linear Regression", value=f"₹{predicted_price:.2f}")
    
    return metrics, predicted_price, rmse, last_actual_price


