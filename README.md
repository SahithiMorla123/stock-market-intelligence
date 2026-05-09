# Stock Market Intelligence System

An AI-powered financial analysis system that collects, processes and analyzes real-time stock market data and news for major companies using Python, RAG, LLaMA 3 and Streamlit.

---

## Project Status — In Progress

This project is actively being built. Below is the current progress:

### Completed

**1. Data Collection**
- Fetched 10 years of real-time stock price data for 5 major companies — Apple, Tesla, Google, Microsoft and Amazon — using Yahoo Finance API
- Collected live financial news articles for all 5 companies using NewsAPI
- Total dataset: 12,575 rows of stock data across all companies

**2. Data Cleaning**
- Removed empty and duplicate rows from the raw dataset
- Filtered and kept only relevant columns — Date, Open, Close, High, Low, Volume, Company
- Saved cleaned dataset ready for analysis

### In Progress

**3. Sentiment Analysis** — using BERT and HuggingFace Transformers to classify news as positive, negative or neutral

**4. Stock Trend Analysis** — identifying price patterns and correlations between news sentiment and stock movement

**5. RAG Pipeline** — integrating ChromaDB vector database and LLaMA 3 via Groq for natural language Q&A over financial data

**6. Interactive Dashboard** — building a Streamlit dashboard with Plotly visualizations

---

## Tech Stack

- **Data Collection:** Python, yfinance, NewsAPI, Requests
- **Data Processing:** Pandas, python-dotenv
- **AI & NLP:** BERT, HuggingFace Transformers, LLaMA 3, RAG, ChromaDB, LangChain
- **Visualization:** Streamlit, Plotly
- **Security:** Environment variables via .env file

---

## Project Structure

- `data_collector.py` — fetches real-time stock data for 5 companies
- `news_collector.py` — fetches live news articles using NewsAPI
- `data_cleaner.py` — cleans and filters raw data

---

## Author

Sahithi Morla
GitHub: https://github.com/SahithiMorla123
Email: morlasaisahithi2031@gmail.com
