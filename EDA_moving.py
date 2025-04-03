import pandas as pd
import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def eda_moving(data):
    st.markdown("<h3 style='color: cyan;'>EDA: Moving Averages</h3>", unsafe_allow_html=True)
    st.write("Moving average smooths out price fluctuations by averaging prices over a set period, reducing noise and helping determine whether a market is trending.")
    
    df_copy = data.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    
    df_copy["MA_30"] = df_copy["Close"].rolling(window=30).mean()
    df_copy["MA_60"] = df_copy["Close"].rolling(window=60).mean()
    df_copy["MA_90"] = df_copy["Close"].rolling(window=90).mean()
    
    fig = go.Figure(data=[
        go.Scatter(x=df_copy.index, y=df_copy["MA_30"], mode="lines", name="30-Day MA", line=dict(color="cyan", dash="dot")),
        go.Scatter(x=df_copy.index, y=df_copy["MA_60"], mode="lines", name="60-Day MA", line=dict(color="yellow", dash="dot")),
        go.Scatter(x=df_copy.index, y=df_copy["MA_90"], mode="lines", name="90-Day MA", line=dict(color="purple", dash="dot"))
    ])
    fig.update_layout(
        title="Moving Averages",
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Stock Price",
        xaxis=dict(rangeslider=dict(visible=True), showline=True, linecolor="white", linewidth=1),
        yaxis=dict(showline=True, linecolor="white", linewidth=1),
        legend_title="Reference",
    )
    st.plotly_chart(fig)
