import streamlit as st
from datetime import date
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="FINTELLI")
st.title('Fintelli')

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
# --------------------------------------------------------------- 
from m2_exponential import m2_exponential
m2_exponential(data)
st.divider()
from future_exponential import future_exponential
future_exponential(data)
st.divider()
# --------------------------------------------------------------- 
from m3_arima import m3_arima
m3_arima(data, order=(6,1,0))
st.divider()
from future_arima import future_arima
future_arima(data, order=(6,1,0))
st.divider()
# --------------------------------------------------------------- 
from m4_sarima import m4_sarima
m4_sarima(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
st.divider()
from future_sarima import future_sarima
future_sarima(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
st.divider()
# --------------------------------------------------------------- 
from m5_random import m5_random
m5_random(data)
st.divider()
from future_random import future_random
future_random(data)
st.divider()
# --------------------------------------------------------------- 
from m6_xg import m6_xg
m6_xg(data, target_column='Close')
st.divider()
from future_xg import future_xg
future_xg(data)
st.divider()
# --------------------------------------------------------------- 
from m7_prophet import m7_prophet
m7_prophet(data)
st.divider()
from future_prophet import future_prophet
future_prophet(data)
st.divider()
# --------------------------------------------------------------- 
from m8_lstm import m8_lstm
m8_lstm(data)
st.divider()
from future_lstm import future_lstm
future_lstm(data)
st.divider()
# --------------------------------------------------------------- 
