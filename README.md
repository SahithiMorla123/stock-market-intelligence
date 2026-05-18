# 📈 Stock Market Intelligence System

An AI-powered stock market analysis system that collects, processes and analyses real-time stock market data and news for any Multinational Company (MNC) worldwide — built using Python, Agentic AI, LangGraph, FinBERT, ChromaDB, RAG and Streamlit.

---

## Project Status

### Completed
- Data Collection — `data_collector.py`
- News Collection — `news_collector.py`
- Data Cleaning — `data_cleaner.py`
- Raw Data Loader — `raw_data_loader.py`
- FinBERT Sentiment Analysis + Technical Indicators — `stock_brain.py`
- Agentic AI Pipeline — `main.py`
- ChromaDB + RAG — integrated in `main.py`

### Remaining
- Streamlit Dashboard — `dashboard.py` (in progress)

---

## What Has Been Built

### 1. Data Collection — `data_collector.py`
- Fetched 10 years of real-time stock market data for multiple MNCs worldwide
- Used yfinance to pull historical OHLCV (Open, High, Low, Close, Volume) data
- Validated each company ticker before fetching to avoid errors
- Saved individual CSV files per company
- Combined all into one master dataset — `all_stocks_data.csv`
- Total dataset: 12,575 rows across all companies

| Tool | Purpose |
|------|---------|
| yfinance | Fetch historical stock market data from Yahoo Finance |
| Pandas | Store, structure and save data as CSV |
| python-dotenv | Load API keys securely from .env file |

---

### 2. News Collection — `news_collector.py`
- Fetched live news articles using NewsAPI
- Collected top 10 latest news articles per company
- Stored title, description, published date and URL for each article
- Saved as CSV for downstream sentiment analysis

| Tool | Purpose |
|------|---------|
| NewsAPI | Fetch real-time news articles for any company |
| Requests | Make HTTP API calls to NewsAPI |
| Pandas | Structure and save news data as CSV |
| python-dotenv | Load NewsAPI key securely |

---

### 3. Data Cleaning — `data_cleaner.py`
- Loaded raw stock data from `all_stocks_data.csv`
- Removed all null and missing values using dropna()
- Kept only the 7 most relevant columns — Date, Open, High, Low, Close, Volume, Company
- Removed duplicate rows using drop_duplicates()
- Reset index for clean row numbering
- Saved cleaned dataset as `clean_stocks_data.csv`

| Tool | Purpose |
|------|---------|
| Pandas | Load, clean, filter and save the dataset |

---

### 4. Raw Data Loader — `raw_data_loader.py`
- Loaded additional stock market dataset from Kaggle
- Successfully loaded 7,163 companies worth of stock data
- Handled file errors and logged failed files

| Tool | Purpose |
|------|---------|
| Pandas | Load and validate Kaggle CSV files |

---

### 5. FinBERT Sentiment Analysis + Technical Indicators — `stock_brain.py`

#### FinBERT Sentiment Analysis
- Integrated FinBERT (ProsusAI/finbert) — a BERT model fine-tuned specifically on stock market news
- Classifies each news article as Positive, Negative or Neutral
- Gives a confidence score for each classification
- Overall market mood calculated from all articles combined
- Sentiment directly influences the agent's BUY / SELL / HOLD recommendation
- Why: Understand how news events (earnings beats, layoffs, lawsuits, product launches) affect stock price movement

#### Technical Indicators
- RSI (Relative Strength Index) — identifies if a stock is overbought (sell signal, RSI above 70) or oversold (buy signal, RSI below 30)
- EMA 20 and EMA 50 (Exponential Moving Average) — identifies short-term and long-term price trends
- MACD (Moving Average Convergence Divergence) — identifies momentum shifts and trend direction changes
- Bollinger Bands — identifies price volatility and potential breakout or reversal signals
- Why: Give investors professional-grade signals beyond just price — the same indicators used by real traders

| Tool | Purpose |
|------|---------|
| HuggingFace Transformers | Load and run the FinBERT model |
| PyTorch | Backend for running FinBERT inference |
| FinBERT (ProsusAI/finbert) | Classify stock market news sentiment |
| yfinance | Fetch price data for indicator calculation |
| Pandas | Calculate RSI, EMA, MACD, Bollinger Bands |

---

### 6. Agentic AI Pipeline — `main.py`
- Built a LangGraph-based Agentic AI system powered by LLaMA 3.3 70B via Groq
- The agent autonomously thinks, decides which tools to use, acts and delivers the answer
- Works for any MNC worldwide — US, Germany, Japan, Korea, India and more
- User simply types a question — agent handles everything automatically
- Human-in-the-Loop integrated at the backend — agent waits for human approval before finalising recommendations

#### Tools Integrated into the Agent

| Tool Name | What It Does |
|-----------|-------------|
| find_ticker | Finds the correct stock ticker symbol for any company name |
| get_stock_price | Fetches live current stock price |
| get_stock_history | Fetches historical OHLCV data for any time period |
| get_company_news | Fetches latest live news articles for the company |
| analyze_sentiment | Runs FinBERT sentiment analysis on fetched news |
| get_technical_analysis | Calculates RSI, EMA, MACD and Bollinger Bands |

| Tool | Purpose |
|------|---------|
| LangGraph | Build the agentic workflow with nodes, edges and state management |
| LangChain | Connect LLM to tools and manage message flow |
| LLaMA 3.3 70B via Groq | The core reasoning brain of the agent |
| Groq | Ultra-fast LLM inference API |
| yfinance | Stock market data inside agent tools |
| NewsAPI | News fetching inside agent tools |

---

### 7. ChromaDB + RAG — integrated in `main.py`
- Stock data and news articles stored as vector embeddings using ChromaDB
- RAG (Retrieval Augmented Generation) enables the agent to answer questions from stored historical data
- Agent retrieves relevant past data before generating answers — giving it memory
- Why: Without RAG the agent only knows what is given right now. With RAG it can answer questions like "How did Apple stock behave during the 2023 earnings season?" by retrieving stored historical context

| Tool | Purpose |
|------|---------|
| ChromaDB | Store and retrieve vector embeddings of stock market and news data |
| RAG | Let the agent retrieve relevant historical context before answering |
| LangChain Embeddings | Convert text data into vectors for ChromaDB storage |

---

## Remaining — Streamlit Dashboard — `dashboard.py`
- Interactive web interface accessible at localhost:8504
- Search bar to find any MNC worldwide — 100+ companies preloaded
- Real-time interactive candlestick chart with Plotly
- Technical indicators displayed on chart — EMA 20/50, RSI, Volume, Bollinger Bands
- Chart reading guide built in — explains green/red candles, RSI zones, volume bars
- Company-specific news with sentiment labels (Positive / Negative / Neutral)
- AI-powered BUY / SELL / HOLD recommendation on demand
- Why: Makes the entire system accessible to non-technical users through a clean UI

| Tool | Purpose |
|------|---------|
| Streamlit | Build and serve the interactive web dashboard |
| Plotly | Render interactive candlestick, RSI and volume charts |
| yfinance | Fetch live stock market data for the dashboard |
| NewsAPI | Fetch company-specific news for the dashboard |
| LLaMA 3.1 8B via Groq | Generate BUY/SELL/HOLD recommendation in dashboard |

---

## Tech Stack

| Area | Tools |
|------|-------|
| Data Collection | Python, yfinance, NewsAPI, Requests |
| Data Processing | Pandas, python-dotenv |
| Sentiment Analysis | FinBERT (ProsusAI/finbert), HuggingFace Transformers, PyTorch |
| Technical Analysis | RSI, EMA 20/50, MACD, Bollinger Bands (Pandas + yfinance) |
| Agentic AI | LangGraph, LangChain, LLaMA 3.3 70B via Groq |
| Vector Database + RAG | ChromaDB, LangChain Embeddings |
| Visualization | Streamlit, Plotly |
| LLM Inference | Groq API |

---

## Project Structure
stock-market-intelligence/
│
├── data_collector.py      — Fetches 10 years of stock market data for MNCs
├── news_collector.py      — Fetches live news articles via NewsAPI
├── data_cleaner.py        — Cleans and filters raw stock dataset
├── raw_data_loader.py     — Loads and validates Kaggle stock dataset
├── stock_brain.py         — FinBERT sentiment + technical indicators
├── main.py                — Agentic AI pipeline (LangGraph + LLaMA + RAG)
├── dashboard.py           — Streamlit interactive dashboard (in progress)
├── all_stocks_data.csv    — Raw combined stock dataset
├── clean_stocks_data.csv  — Cleaned stock dataset
└── .env                   — API keys (GROQ_API_KEY, NEWS_API_KEY)



## Author

Sahithi Morla
- GitHub: https://github.com/SahithiMorla123
- Email: morlasaisahithi2031@gmail.comSonnet 4.6
