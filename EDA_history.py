import streamlit as st
import plotly.graph_objects as go 
import pandas as pd

@st.cache_data
def eda_history(data):
    st.markdown("<h3 style='color: cyan;'>EDA: Time Series Plot</h3>", unsafe_allow_html=True),
    st.write("Historical data plotting for opening and closing prices with opposed to Date, plotted below. Allows for an easier visualisation of the stock price movement over time.")

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
