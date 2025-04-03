import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
    
@st.cache_data
def m5_random(data):
    #st.title("Random Forest Regression Model")
    st.markdown("<h1 style='color: cyan;'>Tree-Based & Ensemble Learning</h1>", unsafe_allow_html=True),
    st.markdown("<h3 style='color: cyan;'>M5: Random Forest Regression Model</h3>", unsafe_allow_html=True),
    st.write("Random Forest is powerful for financial prediction models due to its ability to handle complex, non-linear data and prevent overfitting through ensemble learning. It can rank feature importance, making it useful for identifying key financial factors. Its robustness to missing data and non-linear relationships makes it versatile and accurate for tasks like stock price forecasting and risk assessment.")    
    
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    
    target_column = "Close"
    
    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    fig = go.Figure(data=[
        go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual', line=dict(color='blue')),
        go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Predicted', line=dict(color='orange'))
    ])
    
    fig.update_layout(title='Random Forest',
                    xaxis=dict(title="Date", rangeslider=dict(visible=True)), 
                    yaxis_title=target_column,
                    legend_title='Reference')
    
    st.markdown("`METRIC VALIDATION PLOT`", unsafe_allow_html=True)
    st.plotly_chart(fig)

    st.markdown("`ERROR EVALUATION METRICS`", unsafe_allow_html=True)
    st.subheader("Evaluation Metrics")

    @st.cache_data
    def evaluate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return{
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R^2': r2
        }

    metrics = evaluate_metrics(y_test.values, y_pred)

    #y_test is a Pandas Series,not a df, y_test['Close'] will throw an error 
    #y_pred is a NumPy array, not a DataFrame, y_pred.values will throw an error because y_pred is already an array hence just use y_pred
    # can use below code to know what is what 
    # st.write(type(y_test))  # Is it a DataFrame or Series?
    # st.write(type(y_pred)) 

    blue_text = "color: #3498DB;"
    #st.markdown(f'<p style="{blue_text}"><b>MSE:</b> {metrics["MSE"]:.4f}</p>', unsafe_allow_html=True)
    st.markdown(f"RMSE: The model's predicted prices deviate by around Rs.<span style='{blue_text}'>{metrics['RMSE']:.2f}</span> on average.", unsafe_allow_html=True)
    st.markdown(f"MAE: On average, the model's absolute error in predictions is around <span style='{blue_text}'>{metrics['MAE']:.2f}</span>.", unsafe_allow_html=True)
    st.markdown(f"MAPE: The model's predictions have an average error of <span style='{blue_text}'>{metrics['MAPE']:.2f}%</span> relative to actual values.", unsafe_allow_html=True)
    st.markdown(f"R^2: The <span style='{blue_text}'>R² value of {metrics['R^2']:.4f}</span> indicates that the model explains <span style='{blue_text}'>{metrics['R^2'] * 100:.2f}%</span> of the variance in the target variable. Higher values (closer to 1) indicate better fit.", unsafe_allow_html=True)

    predicted_price = y_pred[-1] if len(y_pred) > 0 else None
    st.markdown("`CLOSING PRICE PREDICTION FOR THE DAY`", unsafe_allow_html=True)
    st.metric(label="Random Forest", value=f"₹{predicted_price:.2f}" if predicted_price is not None else "N/A")
