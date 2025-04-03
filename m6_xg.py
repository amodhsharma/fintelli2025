import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

@st.cache_data
def m6_xg(data, target_column='Close'):
    st.markdown("<h3 style='color: cyan;'>M6: XG Boost</h3>", unsafe_allow_html=True)
    st.write("XGBoost is a gradient boosting framework that uses decision trees to create an ensemble model. It is known for its speed, performance, and ability to handle large datasets with high dimensionality.")
    
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # model.save_model("xgboost_model.json")  # Save model for future use #saves a file

    y_pred = model.predict(X_test)

    fig = go.Figure(data=[
        go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual', line=dict(color='blue')),
        go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Predicted', line=dict(color='orange'))
    ])
    
    fig.update_layout(title='XGBoost Model',
                      xaxis=dict(title="Date", rangeslider=dict(visible=True)), 
                      yaxis_title=target_column,
                      legend_title='Reference')

    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)
    st.plotly_chart(fig)

    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        'R^2': r2_score(y_test, y_pred)
    }

    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)
    st.subheader("Evaluation Metrics")

    blue_text = "color: #3498DB;"
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    predicted_price = y_pred[-1] if len(y_pred) > 0 else None
    st.markdown("`CLOSING PRICE PREDICTION FOR THE DAY`", unsafe_allow_html=True)
    st.metric(label="XGBoost", value=f"₹{predicted_price:.2f}" if predicted_price is not None else "N/A")
