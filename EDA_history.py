import streamlit as st
import plotly.graph_objects as go 
import pandas as pd

@st.cache_data
def eda_history(data):
    st.markdown("<h3 style='color: cyan;'>EDA: Time Series Plot</h3>", unsafe_allow_html=True),
    st.write("Time series plots are crucial for financial advisors and stock market analysts to visualize stock price movements over time. By identifying trends, seasonality, and irregular fluctuations, advisors can make data-driven investment decisions, predict future stock performance, and assess market behavior effectively.")
    st. write("The below Graph shows the stock open and close prices over time.")
    
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)

    # st.write(data.head())
    # st.write(data.tail())
    # st.write(data.dtypes)
    # st.write(data.shape)

    st.markdown("`EXPLORATORY DATA ANALYSIS`", unsafe_allow_html=True)

    fig = go.Figure(data=[  #this remains as data
        go.Scatter(x=df_copy.index, y=df_copy['Open'], mode='lines', name="stock_open", line=dict(color="green")),
        go.Scatter(x=df_copy.index, y=df_copy['Close'], mode='lines', name="stock_close", line=dict(color="red"))
    ])

    fig.update_layout(
        autosize=True,
        title="Time Series Plot",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title='Reference',
        xaxis=dict(
            rangeslider=dict(visible=True),
            showline=True,  # Show x-axis line
            linecolor="white",  # Black x-axis line
            linewidth=1  # Make it bold
        ),
        yaxis=dict(
            showline=True,  # Show y-axis line
            linecolor="white",  # Black y-axis line
            linewidth=1  # Make it bold
        ),
    )
    st.plotly_chart(fig)
