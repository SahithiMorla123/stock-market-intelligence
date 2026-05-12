# Stock Market Intelligence System

An AI-powered financial analysis system that collects, processes and analyses 
real-time stock market data and news for any Multinational Company (MNC) worldwide 
— built using Python, Agentic AI, LangGraph, FinBERT and Streamlit.

---

## What Has Been Built

### 1. Data Collection — `data_collector.py`
- Fetched **10 years** of real-time stock price data for multiple Multinational Companies (MNCs)
- Used **yfinance** to pull historical OHLCV data for each company
- Validated each company ticker before fetching
- Saved individual CSV files per company
- Combined all into one master dataset `all_stocks_data.csv`
- **Total dataset: 12,575 rows** across all companies
- **Tools used:** yfinance, Pandas, python-dotenv

### 2. News Collection — `news_collector.py`
- Fetched live financial news articles using **NewsAPI**
- Collected top 10 latest news articles per company
- Stored title, description, published date and URL for each article
- Saved as CSV for sentiment analysis
- **Tools used:** Requests, Pandas, NewsAPI, python-dotenv

### 3. Data Cleaning — `data_cleaner.py`
- Loaded raw stock data from `all_stocks_data.csv`
- Removed all null and missing values using `dropna()`
- Kept only the 7 most relevant columns — **Date, Open, High, Low, Close, Volume, Company**
- Removed duplicate rows using `drop_duplicates()`
- Reset index for clean row numbering
- Saved cleaned dataset as `clean_stocks_data.csv`
- **Tools used:** Pandas

### 4. Raw Data Loader — `raw_data_loader.py`
- Loaded additional stock market dataset from Kaggle
- Successfully loaded **7,163 companies** worth of stock data
- Handled file errors and logged failed files
- **Tools used:** Pandas

### 5. Agentic AI Pipeline — `main.py`
- Built a **LangGraph-based Agentic AI** system powered by **LLaMA 3.3 70B via Groq**
- The agent autonomously thinks, decides which tools to use, acts and delivers answers
- Works for **any Multinational Company (MNC) worldwide** — US, Germany, Japan, Korea, India and more
- User simply types a question — agent handles everything automatically
- **Tools integrated into the agent:**
  - `find_ticker` — finds correct stock ticker for any company name
  - `get_stock_price` — fetches live current stock price
  - `get_stock_history` — fetches historical data for any period
  - `get_company_news` — fetches latest live news articles
  - `analyze_sentiment` — runs FinBERT sentiment analysis on news
  - `get_technical_analysis` — calculates RSI, EMA, MACD, Bollinger Bands
- **Tools used:** LangGraph, LangChain, LLaMA 3.3 70B, Groq, yfinance, NewsAPI

### 6. FinBERT Sentiment Analysis + Technical Indicators — `stock_brain.py`

**FinBERT Sentiment Analysis:**
- Integrated **FinBERT (ProsusAI/finbert)** — a BERT model specifically fine-tuned on financial news
- Classifies each news article as **Positive, Negative or Neutral**
- Gives a confidence score for each classification
- Overall market mood calculated from all articles combined
- Sentiment directly influences the agent's buy/sell/hold recommendation
- **Motive:** Understand how news and market events (layoffs, earnings, lawsuits) affect stock prices
- **Tools used:** HuggingFace Transformers, PyTorch, FinBERT

**Technical Indicators:**
- **RSI (Relative Strength Index)** — identifies if stock is overbought (sell signal) or oversold (buy signal)
- **EMA 20 & 50 (Exponential Moving Average)** — identifies short and long term price trends
- **MACD (Moving Average Convergence Divergence)** — identifies momentum and trend direction changes
- **Bollinger Bands** — identifies price volatility and potential breakout or reversal signals
- **Motive:** Give investors professional-grade signals beyond just price — the same indicators used by real traders
- **Tools used:** yfinance, Pandas

---

## What Is Coming Next

### 7. LSTM Price Prediction
- Training an **LSTM (Long Short-Term Memory)** model on 10 years of stock data
- Dataset will be enriched with FinBERT sentiment scores and technical indicators
- Predicting stock price trend for **next 5-7 days**
- Combining price prediction + sentiment + technical indicators for stronger recommendations
- **Motive:** Give users a forward-looking view, not just historical analysis

### 8. ChromaDB Vector Database + RAG
- Storing stock data and news as vectors using **ChromaDB**
- Enabling **RAG (Retrieval Augmented Generation)** so the agent can answer questions from stored historical data
- **Motive:** Give the agent memory so it can answer complex historical queries

### 9. Streamlit Dashboard
- Interactive web interface with **search bar** and **calendar/date picker**
- Real-time **interactive stock charts** with Plotly
- Sentiment score visualization
- Technical indicator display
- AI-powered **buy/sell/hold recommendation**
- **Motive:** Make the system accessible to non-technical users through a beautiful UI

---

## Tech Stack

| Area | Tools |
|---|---|
| Data Collection | Python, yfinance, NewsAPI, Requests |
| Data Processing | Pandas, python-dotenv |
| Agentic AI | LangGraph, LangChain, LLaMA 3.3 70B via Groq |
| Sentiment Analysis | FinBERT (ProsusAI/finbert), HuggingFace Transformers, PyTorch |
| Technical Analysis | RSI, EMA, MACD, Bollinger Bands (via Pandas + yfinance) |
| Vector Database | ChromaDB (coming soon) |
| Prediction Model | LSTM (coming soon) |
| Visualization | Streamlit, Plotly (coming soon) |

---

## Project Structure

- `data_collector.py` — fetches 10 years of stock data for multiple MNCs
- `news_collector.py` — fetches live news articles using NewsAPI
- `data_cleaner.py` — cleans and filters raw stock dataset
- `raw_data_loader.py` — loads and validates Kaggle stock dataset
- `stock_brain.py` — FinBERT sentiment analysis + technical indicators
- `main.py` — Agentic AI pipeline with LangGraph and LLaMA 3.3

---

## Author

**Sahithi Morla**
GitHub: https://github.com/SahithiMorla123
Email: morlasaisahithi2031@gmail.com
