import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st

@st.cache_data
def extraction_m6(data, target_column='Close'):
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    # Train XGBoost model
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Get last actual and predicted price
    last_actual_price = y_test.iloc[-1]
    last_predicted_price = y_pred[-1] if len(y_pred) > 0 else None

    return rmse, last_actual_price, last_predicted_price
