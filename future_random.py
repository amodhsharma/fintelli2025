import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

@st.cache_data
def future_random(data):
    st.markdown("<h3 style='color: cyan;'>M5.1: Future Predictions using Random Forest</h3>", unsafe_allow_html=True)

    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    target_column = "Close"  # Target variable

    # Splitting into features (X) and target (y)
    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]

    # Train model using the entire dataset
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Generate future dates for the next month
    last_date = df_copy.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')  # 'B' for business days

    # Use last available feature values as a base for future predictions
    last_features = X.iloc[-1].values.reshape(1, -1)

    future_predictions = []
    
    for _ in range(30):
        predicted_price = model.predict(last_features)[0]
        future_predictions.append(predicted_price)

        # Update features for next iteration (assuming features do not drastically change)
        last_features = np.append(last_features[:, 1:], predicted_price).reshape(1, -1)

    # Convert predictions into DataFrame
    future_df = pd.DataFrame({target_column: future_predictions}, index=future_dates)

    # Combine last 100 actual data points with predicted values
    recent_actuals = df_copy.iloc[-100:][target_column]
    
    fig = go.Figure(data=[
        go.Scatter(x=recent_actuals.index, y=recent_actuals, mode='lines', name='Actual (Last 100 Days)', line=dict(color='blue')),
        go.Scatter(x=future_df.index, y=future_df[target_column], mode='lines', name='Predicted (Next 30 Days)', line=dict(color='orange', dash='dash'))
    ])
    
    fig.update_layout(
        title="Future Predictions using Random Forest",
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis_title="Stock Closing Price",
        legend_title="Legend"
    )

    st.markdown("`PRICE OF NEXT 30 DAYS FORCAST`", unsafe_allow_html=True)
    st.plotly_chart(fig)
    

