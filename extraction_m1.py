import pandas as pd
import numpy as np 

@st.cache_data
def extraction_m1(data):
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    target_column = "Close"

    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    last_actual_price = y_test.iloc[-1]
    last_predicted_price = y_pred[-1]  # Extract last predicted value

    return rmse, last_actual_price, last_predicted_price  
