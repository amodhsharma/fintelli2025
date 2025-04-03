import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

@st.cache_data
def extraction_m3(data, order=(6,1,0)):
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    train_size = int(len(df_copy) * 0.85)
    train, test = df_copy[:train_size], df_copy[train_size:]

    # Train ARIMA model
    model = ARIMA(train['Close'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))

    # Compute RMSE
    def evaluate_forecast(actual, predicted):
        min_len = min(len(actual), len(predicted))
        actual, predicted = actual[:min_len], predicted[:min_len]
        return np.sqrt(mean_squared_error(actual, predicted))

    rmse = evaluate_forecast(test['Close'].values, forecast.values)

    # Get last actual and predicted price
    last_actual_price = test['Close'].iloc[-1]
    last_predicted_price = forecast.iloc[-1] if not forecast.empty else None

    return rmse, last_actual_price, last_predicted_price
