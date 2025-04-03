import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error

@st.cache_data
def extraction_m2(data):
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    train_size = int(len(data) * 0.85)
    train, test = data[:train_size], data[train_size:]

    # Train the model
    model_single = SimpleExpSmoothing(train['Close']).fit(smoothing_level=0.2, optimized=True)
    forecast_single = model_single.forecast(len(test))

    # Compute RMSE
    def evaluate_forecast(actual, predicted):
        min_len = min(len(actual), len(predicted))
        actual, predicted = actual[:min_len], predicted[:min_len]
        return np.sqrt(mean_squared_error(actual, predicted))

    rmse = evaluate_forecast(test['Close'].values, forecast_single)

    # Get last actual and predicted price
    last_actual_price = test['Close'].iloc[-1]
    last_predicted_price = forecast_single.iloc[-1] if not forecast_single.empty else None

    return rmse, last_actual_price, last_predicted_price
