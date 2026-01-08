# 📈 Smart Stock Insight

An AI-powered stock market analysis and prediction platform providing technical indicators, next-day price prediction, 7-day LSTM forecasting, portfolio comparison, and sentiment analysis.

## 📂 Directory Structure
smart-stock-insight/
├── README.md                   # Project documentation 
├── app.py                      # Streamlit web application
├── sentiment.py                # News sentiment analysis module
├── requirements.txt            # Python dependencies
├── venv/                       # Virtual environment
└── screenshots/
    ├── Market_Analysis.png
    ├── Price_Prediction.png
    ├── Sentiment_Analysis.png
    └── Portfolio_Comparison.png                    

## 🚀 Quick Start

### 1. Clone Repository
git clone https://github.com/rajeshwarikanawat/smart-stock-insight.git
cd smart-stock-insight

### 2. Set up Virtual Environment
python3.10 -m venv venv
venv\Scripts\activate        


### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run Streamlit App
streamlit run app.py

➡️ Access at http://localhost:8501

## 🔄 Data Processing & Features

### Stock Selection
- Dynamic input of ticker symbol, start, and end dates
- Historical data retrieved using yfinance

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- 20-day and 50-day moving averages
- Bullish/Bearish trend detection

### Price Prediction
- Next-Day Forecast using Linear Regression
- 7-Day Forecast using LSTM neural network
- Data normalized via MinMaxScaler

### Portfolio Comparison
- Multi-stock selection
- Calculates returns & cumulative performance
- Interactive line charts via Plotly

### Sentiment Analysis
- Fetches news articles
- Analyzes market sentiment using TextBlob or newspaper3k (if configured)

## 🤖 Machine Learning Models

### Linear Regression (Next-Day Price Prediction)
- Input: Close price
- Output: Predicted next day Close
- Lightweight and interpretable

### LSTM Neural Network (7-Day Forecast)
- Input: 60 previous closing prices
- Output: Next 7 closing prices
- Scaled with MinMaxScaler for numerical stability
- Architecture:
```python
Sequential([
    LSTM(50, return_sequences=True, input_shape=(60,1)),
    LSTM(50),
    Dense(1)
])
```
- Loss: MSE | Optimizer: Adam

## 🌐 Streamlit Web Application

### Interactive Features
- Responsive sidebar for ticker and date selection
- Real-time technical indicator charts
- Next-day price forecast & 7-day LSTM forecast visualization
- Portfolio cumulative return comparison
- Optional news sentiment analysis

### Example Metrics Display
- Last Close: $150.23
- Day High: $153.10
- Day Low: $148.90

### Portfolio Performance
- Select multiple stocks
- Displays cumulative returns line chart
- Supports AAPL, MSFT, TSLA, GOOG, AMZN

## 🛠️ Tech Stack

| Component        | Technologies                     |
|-----------------|---------------------------------|
| Frontend        | Streamlit                        |
| ML              | TensorFlow, Scikit-learn         |
| Data            | Pandas, NumPy                    |
| Visualization   | Plotly, Matplotlib, Seaborn      |
| News Sentiment  | newspaper3k, TextBlob            |
| Environment     | Python 3.10, virtualenv          |

## 📌 Key Notes
- **Data Source:** Yahoo Finance via yfinance
- **Reproducibility:** Seed locking for LSTM and ML models
- **Error Handling:** Empty data handling for invalid ticker symbols
- **Scalability:** Can fetch multi-year historical stock data

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author
**Rajeshwari Kanawat**  
🔗 Repository: [Smart Stock Insight](https://github.com/rajeshwarikanawat/smart-stock-insight)




