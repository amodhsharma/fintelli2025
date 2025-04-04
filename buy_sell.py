# buy_sell.py

from extraction_m1 import extraction_m1
from extraction_m2 import extraction_m2
from extraction_m3 import extraction_m3
from extraction_m4 import extraction_m4
from extraction_m5 import extraction_m5
from extraction_m6 import extraction_m6
from extraction_m7 import extraction_m7
from extraction_m8 import extraction_m8
import streamlit as st

@st.cache_data
def buy_sell_decision(actual_price, predicted_price, rmse):
    if predicted_price > actual_price + rmse:
        return "BUY 📈"
    elif predicted_price < actual_price - rmse:
        return "SELL 📉"
    else:
        return "HOLD ⏳"

@st.cache_data
def process_models(data):
    models = {
        "Linear Regression": extraction_m1(data),
        "Exponential Smoothening": extraction_m2(data),
        "Arima": extraction_m3(data),
        "Sarima": extraction_m4(data),
        "Random Forest": extraction_m5(data),
        "XG Boost": extraction_m6(data),
        "Prophet": extraction_m7(data),
        "LSTM": extraction_m8(data)
    }

    decisions = {}
    for model_name, (rmse, actual, predicted) in models.items():
        decision = buy_sell_decision(actual, predicted, rmse)
        decisions[model_name] = {
            "RMSE": rmse,
            "Actual": actual,
            "Predicted": predicted,
            "Decision": decision
        }
    
    return decisions
