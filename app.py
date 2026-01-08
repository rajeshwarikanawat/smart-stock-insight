import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

from sentiment import analyze_sentiment
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


st.set_page_config(
    page_title="Smart Stock Insight",
    layout="wide"
)

st.title("Smart Stock Insight PLatform")
st.caption("AI Powered Stock Market Analysis & Prediction")

st.sidebar.header("Stock Selection")

ticker = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date= st.sidebar.date_input("End Date", pd.to_datetime("today"))

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)

    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    return df

df = load_data(ticker,start_date, end_date)

if df.empty:
    st.error("No data found. Check stock symbol")
    st.stop()


col1,col2,col3 = st.columns(3)
col1.metric("Last Close",f"${float(df['Close'].iloc[-1]):.2f}")
col2.metric("Day High", f"${float(df['High'].iloc[-1]):.2f}")
col3.metric("Day Low", f"${float(df['Low'].iloc[-1]):.2f}")


tab1, tab2, tab3,tab4 = st.tabs([
        "Market Analysis",
        "Price Prediction",
        "Portfolio Comparison",
        "News Sentiment"
    ])

with tab1:
    st.subheader("Stock Price Chart")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price Movement"
        ))

    fig.update_layout(
        title="Stock Price Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        height=500,
        xaxis_rangeslider_visible=False
        )

    st.plotly_chart(fig, use_container_width=True)

        
    close_series = df['Close'].astype(float)

    df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()

    macd=MACD(close=close_series)
    df['MACD'] = macd.macd()
    col1,col2 = st.columns(2)
    fig_rsi = go.Figure()


    fig_rsi.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        name='RSI Value'
        ))

    fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Oversold")

    fig_rsi.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI Value",
        legend_title="Indicators"
        )

    col1.plotly_chart(fig_rsi, use_container_width=True)

    fig_macd = go.Figure()

    fig_macd.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        mode='lines',
        name='MACD Line'
        ))

    fig_macd.update_layout(
        title="MACD Indicator",
        xaxis_title="Date",
        yaxis_title="MACD Value",
        legend_title="MACD Components"
        )

    col2.plotly_chart(fig_macd, use_container_width=True)




    ma20 = df['Close'].rolling(20).mean()
    ma50 = df['Close'].rolling(50).mean()

    if ma20.iloc[-1] > ma50.iloc[-1]:
        st.success("Market Trend: Bullish")
    else:
        st.error("Market Trend: Bearish")


with tab2:
    st.subheader("Next-Day Price Prediction (ML)")
    df_pred = df[["Close"]].copy()
    df_pred['Target']=df_pred['Close'].shift(-1)
    df_pred.dropna(inplace=True)


    x = df_pred[['Close']]
    y= df_pred['Target']

    ml_model = LinearRegression()
    ml_model.fit(x,y)

    next_price = ml_model.predict([[df_pred['Close'].iloc[-1]]])
    st.metric("Predicted Next Close",f"${next_price[0]:.2f}")

    st.divider()
    st.subheader("7-Day Price Forecast (LSTM)")

    def create_lstm_data(data, steps=60):
        x,y=[],[]
        for i in range(steps, len(data)):
            x.append(data[i-steps:i])
            y.append(data[i])
        return np.array(x), np.array(y)
    if st.button("Run LSTM Forecast"):
        close_prices = df[['Close']].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_prices)
        x_lstm,y_lstm = create_lstm_data(scaled)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_lstm.shape[1],1)),
            LSTM(50),
            Dense(1)

            ])

        model.compile(optimizer="adam",loss="mse")
        model.fit(x_lstm,y_lstm, epochs=5, batch_size=32, verbose=0)

        last_60 = scaled[-60:].reshape(1,60,1)
        future=[]


        for _ in range(7):
            pred = model.predict(last_60, verbose=0)
            future.append(pred[0][0])
            last_60 = np.concatenate(
                [last_60[:, 1:, :], pred.reshape(1, 1, 1)],
                axis=1
            )

        future_prices = scaler.inverse_transform(
                np.array(future).reshape(-1,1)
        )

        future_dates = pd.date_range(
        start=df.index[-1],
        periods=7,
        freq='B'
        )

        fig_lstm = go.Figure()

        fig_lstm.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices.flatten(),
            mode='lines+markers',
            name='LSTM Forecast'
            ))

        fig_lstm.update_layout(
            title="7-Day Stock Price Forecast (LSTM)",
            xaxis_title="Date",
            yaxis_title="Predicted Price",
            legend_title="Forecast"
            )

        st.plotly_chart(fig_lstm, use_container_width=True)

        st.success("7-Day Forecast Generated")


with tab3:
    st.subheader("Portfolio Performance")
    stocks = st.multiselect(
        "Select Stocks",
        ["AAPL","MSFT","TSLA","GOOG","AMZN"],
        default=["AAPL","MSFT"]
    )
    if stocks:
        portfolio = pd.DataFrame()

        for s in stocks:
            data = yf.download(s, start=start_date, end=end_date)
            portfolio[s] = data['Close']

        returns = portfolio.pct_change()
        cumulative = (1 + returns).cumprod()

        fig_portfolio = go.Figure()
        for stock in cumulative.columns:
            fig_portfolio.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative[stock],
                mode='lines',
                name=stock
            ))
        fig_portfolio.update_layout(
            title="Portfolio Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            legend_title="Stocks"

        )
        
        st.plotly_chart(fig_portfolio, use_container_width=True)

          
        st.success("Portfolio comparison completed")
with tab4:
    st.subheader("Stock News Sentiment Analysis")
    news_urls = [
        f"https://finance.yahoo.com/quote/{ticker}",
        f"https://www.marketwatch.com/investing/stock/{ticker}"

    ]
    sentiment_score = analyze_sentiment(news_urls)
    st.metric("Sentiment Score", round(sentiment_score,3))

    if sentiment_score >0.1:
        st.success("Positive Market Sentiment")
    elif sentiment_score < -0.1:
        st.error("Negative Market Sentiment")
    else:
        st.info("Neutral Market Sentiment")     
