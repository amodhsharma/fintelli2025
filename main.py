import streamlit as st
from datetime import date
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="FINTELLI")
st.title('Fintelli')
st.write("Your very own technical analysis tool for stock market prediction")

# ---------------------------------------------------------------

from stocks import stocks
ticker = st.selectbox("`USER INPUT` Select Stock Ticker:", stocks) 

import yfinance as yf
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, "2015-01-01", date.today().strftime("%Y-%m-%d"))
    data.index = pd.to_datetime(data.index) # Ensure 'Date' is datetime and set as index
    data = data.sort_index()  # Ensure chronological order
    data.reset_index(inplace=True)  #resetting index to get date as main column
    return data

data = load_data(ticker)    #loading data
st.markdown(f"✅ **Showing data for `{ticker}`**", unsafe_allow_html=True)

# ---------------------------------------------------------------

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)  # Drop the second level

st.markdown("`PREPROCESSING` Raw data - From the beginning ", unsafe_allow_html=True)
st.write(data.head())
st.markdown("`PREPROCESSING` Raw data - Towards the end ", unsafe_allow_html=True)
st.write(data.tail())
st.divider()

# ---------------------------------------------------------------
from EDA_history import eda_history
eda_history(data)
st.divider()
# ---------------------------------------------------------------
from EDA_volatality import eda_volatility
eda_volatility(data)
st.divider()
# ---------------------------------------------------------------
from EDA_moving import eda_moving
eda_moving(data)
st.divider()
# --------------------------------------------------------------- 
from m1_linear import process_and_train_model
process_and_train_model(data)
st.divider()

from future_linear import future_linear
future_linear(data)
st.divider()

# from extraction_m1 import extraction_m1
# rmse, last_actual_price, last_predicted_price = extraction_m1(data)
# st.write(f"RMSE: {rmse:.2f}")
# st.write(f"Last Actual Price: ₹{last_actual_price:.2f}")
# st.write(f"Last Predicted Price: ₹{last_predicted_price:.2f}")
# --------------------------------------------------------------- 
from m2_exponential import m2_exponential
m2_exponential(data)
st.divider()

from future_exponential import future_exponential
future_exponential(data)
st.divider()

# from extraction_m2 import extraction_m2
# rmse, last_actual_price, last_predicted_price = extraction_m2(data)
# st.write(f"RMSE: {rmse:.2f}")
# st.write(f"Last Actual Price: ₹{last_actual_price:.2f}")
# st.write(f"Last Predicted Price: ₹{last_predicted_price:.2f}")
# --------------------------------------------------------------- 
from m3_arima import m3_arima
m3_arima(data, order=(6,1,0))
st.divider()

from future_arima import future_arima
future_arima(data, order=(6,1,0))
st.divider()

# from extraction_m3 import extraction_m3
# rmse, last_actual_price, last_predicted_price = extraction_m3(data)
# st.write(f"RMSE: {rmse:.2f}")
# st.write(f"Last Actual Price: ₹{last_actual_price:.2f}")
# st.write(f"Last Predicted Price: ₹{last_predicted_price:.2f}")
# --------------------------------------------------------------- 
from m4_sarima import m4_sarima
m4_sarima(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
st.divider()

from future_sarima import future_sarima
future_sarima(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
st.divider()

# from extraction_m4 import extraction_m4
# rmse, last_actual_price, last_predicted_price = extraction_m4(data)
# st.write(f"RMSE: {rmse:.2f}")
# st.write(f"Last Actual Price: ₹{last_actual_price:.2f}")
# st.write(f"Last Predicted Price: ₹{last_predicted_price:.2f}")
# --------------------------------------------------------------- 
from m5_random import m5_random
m5_random(data)
st.divider()

from future_random import future_random
future_random(data)
st.divider()

# from extraction_m5 import extraction_m5
# rmse, last_actual_price, last_predicted_price = extraction_m5(data)
# st.write(f"RMSE: {rmse:.2f}")
# st.write(f"Last Actual Price: ₹{last_actual_price:.2f}")
# st.write(f"Last Predicted Price: ₹{last_predicted_price:.2f}")
# --------------------------------------------------------------- 
from m6_xg import m6_xg
m6_xg(data, target_column='Close')
st.divider()

from future_xg import future_xg
future_xg(data)
st.divider()

# from extraction_m6 import extraction_m6
# rmse, last_actual_price, last_predicted_price = extraction_m6(data)
# st.write(f"RMSE: {rmse:.2f}")
# st.write(f"Last Actual Price: ₹{last_actual_price:.2f}")
# st.write(f"Last Predicted Price: ₹{last_predicted_price:.2f}")
# --------------------------------------------------------------- 
from m7_prophet import m7_prophet
m7_prophet(data)
st.divider()

from future_prophet import future_prophet
future_prophet(data)
st.divider()

# from extraction_m7 import extraction_m7
# rmse, last_actual_price, last_predicted_price = extraction_m7(data)
# st.write(f"RMSE: {rmse:.2f}")
# st.write(f"Last Actual Price: ₹{last_actual_price:.2f}")
# st.write(f"Last Predicted Price: ₹{last_predicted_price:.2f}")
# --------------------------------------------------------------- 
from m8_lstm import m8_lstm
m8_lstm(data)
st.divider()

from future_lstm import future_lstm
future_lstm(data)
st.divider()
# --------------------------------------------------------------- 

from buy_sell import process_models
decisions = process_models(data)

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame.from_dict(decisions, orient='index')
df.index.name = "Models" #first column is model names

df["Price Difference"] = df["Predicted"] - df["Actual"]
df.rename(columns={"RMSE": "Calculated RMSE"}, inplace=True)

# Format numeric columns (optional, but makes it more readable)
df["Calculated RMSE"] = df["Calculated RMSE"].apply(lambda x: f"{x:.2f}")
df["Actual"] = df["Actual"].apply(lambda x: f"₹{x:.2f}")
df["Predicted"] = df["Predicted"].apply(lambda x: f"₹{x:.2f}")
df["Price Difference"] = df["Price Difference"].apply(lambda x: f"₹{x:.2f}")

st.markdown("Model Comparison Table with RMSE and Price Difference")
st.dataframe(df, use_container_width=True)

st.divider()
# --------------------------------------------------------------- 
#implementing final buy sell decision
from buy_sell_decision import weighted_decision
final_decision, score = weighted_decision(data)

# Display in Streamlit
st.markdown("`Final Decision based on technical analysis`")
st.metric(label="", value=final_decision)
st.write(f"Weighted Score: `{score:.4f}`")

# --------------------------------------------------------------- 

with st.container():
    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: gray;'>
        April 2025 Fintelli | Built with attention to detail and devotion using Streamlit | 
        <a href="https://github.com/amodhsharma/fintelli2025" target="_blank" style="color: gray; text-decoration: none;">
            Source Code
        </a>
    </p>
    """, unsafe_allow_html=True)

