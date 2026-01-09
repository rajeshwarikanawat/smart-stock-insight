# 📈 Smart Stock Insight

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Deployed%20With-Streamlit-ff4b4b)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Smart Stock Insight** is an end-to-end AI-powered stock market analysis and prediction platform that integrates technical analysis, machine learning, deep learning, portfolio analytics, and news sentiment analysis into a single interactive web application.

The project demonstrates a complete real-world financial analytics pipeline — from data ingestion to deployment — built using Python and Streamlit.

---

## 📂 Directory Structure

```plaintext
smart-stock-insight/
├── README.md                   # Project documentation
├── app.py                      # Streamlit web application (core logic)
├── sentiment.py                # News sentiment analysis module
├── requirements.txt            # Python dependencies
├── venv/                       # Virtual environment (local)
└── screenshots/                # Application screenshots
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <https://github.com/rajeshwarikanawat/smart-stock-insight.git>
cd smart-stock-insight
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate       # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

➡️ Open in browser: **http://localhost:8501**

---

## 🔄 System Architecture & Data Flow

```plaintext
Yahoo Finance Data
        ↓
Data Cleaning & Validation
        ↓
Technical Indicators (RSI, MACD, MAs)
        ↓
Trend Classification
        ↓
ML Prediction (Linear Regression)
        ↓
DL Forecasting (LSTM)
        ↓
Portfolio Analysis
        ↓
News Sentiment Scoring
```

---

## 📊 Market Analysis Module

### Data Ingestion
Stock price data is fetched dynamically from Yahoo Finance using `yfinance`.

```python
df = yf.download(ticker, start=start_date, end=end_date)
df.dropna(inplace=True)
```

### Candlestick Visualization
Interactive candlestick charts are rendered using Plotly.

```python
go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)
```

### Technical Indicators

**RSI (Relative Strength Index)**
```python
df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
```

**MACD**
```python
macd = MACD(close=df['Close'])
df['MACD'] = macd.macd()
```

**Moving Averages & Trend Detection**
```python
ma20 = df['Close'].rolling(20).mean()
ma50 = df['Close'].rolling(50).mean()

trend = "Bullish" if ma20.iloc[-1] > ma50.iloc[-1] else "Bearish"
```

---

## 🤖 Price Prediction Module

### 1️⃣ Next-Day Price Prediction (Linear Regression)

A lightweight and interpretable ML model predicts the next trading day's closing price.

```python
df_pred['Target'] = df_pred['Close'].shift(-1)
model = LinearRegression()
model.fit(X, y)
next_day_price = model.predict([[latest_close]])
```

- **Input**: Current close price  
- **Output**: Predicted next-day close  

---

### 2️⃣ 7-Day Price Forecast (LSTM Neural Network)

A deep learning model learns historical time-series patterns to forecast short-term prices.

```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
```

**Key Configuration**
- Lookback window: 60 days
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Forecast horizon: 7 business days

---

## 📈 Portfolio Comparison Module

Allows users to compare the cumulative performance of multiple stocks.

```python
returns = portfolio.pct_change()
cumulative_returns = (1 + returns).cumprod()
```

**Features**
- Multi-stock selection
- Relative performance visualization
- Diversification analysis

---

## 📰 News Sentiment Analysis Module

Market sentiment is derived from financial news articles related to the selected stock.

```python
article = Article(url)
article.download()
article.parse()
polarity = TextBlob(article.text).sentiment.polarity
```

**Sentiment Interpretation**
- **Positive**: polarity > 0.1  
- **Negative**: polarity < -0.1  
- **Neutral**: otherwise  

---

## 🌐 Streamlit Web Application

### Interface Design
- Sidebar-based ticker & date selection
- Tab-based navigation:
  - Market Analysis
  - Price Prediction
  - Portfolio Comparison
  - News Sentiment

### Visualizations
- Candlestick price charts
- RSI & MACD indicator plots
- LSTM forecast curves
- Portfolio cumulative return comparison

---

## 🛠️ Tech Stack

| Layer            | Technologies                          |
|------------------|---------------------------------------|
| Frontend         | Streamlit                             |
| Data             | yfinance, Pandas, NumPy               |
| Machine Learning | Scikit-learn                          |
| Deep Learning    | TensorFlow (LSTM)                     |
| Indicators       | ta-lib                                |
| Visualization    | Plotly                                |
| NLP              | TextBlob, newspaper3k                 |
| Environment      | Python 3.10                           |

---

## 📌 Key Highlights

- Uses **real market data**
- Combines **technical analysis + ML + DL**
- Fully interactive and modular
- Suitable for academic, portfolio, and demo use cases
- Easily extensible to additional indicators or models

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Rajeshwari Kanawat**

---

> *A unified platform combining financial analytics, predictive modeling, and interactive visualization for smarter market insights.*
