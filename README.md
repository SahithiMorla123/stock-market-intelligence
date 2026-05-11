# Stock Market Intelligence System

An AI-powered financial analysis system that collects, processes and analyses real-time stock market data and news for major companies — built using Python, Agentic AI, LangGraph, LLaMA 3.1 and Streamlit.

---

## What Has Been Built

### 1. Data Collection
- Fetched 10 years of real-time stock price data for 5 major companies — Apple, Tesla, Google, Microsoft and Amazon — using yfinance
- Collected live financial news articles for all 5 companies
- Total dataset: 12,575 rows of stock data across all companies

### 2. Data Cleaning
- Removed empty and duplicate rows from the raw dataset
- Filtered and kept only relevant columns — Date, Open, Close, High, Low, Volume, Company
- Saved cleaned dataset ready for analysis

### 3. Agentic AI Pipeline
- Built a LangGraph-based Agentic AI system powered by LLaMA 3.1 via Groq
- The agent autonomously retrieves real-time stock prices, fetches live news and delivers investment recommendations to the user
- Tools integrated into the agent: stock price retrieval and live news fetching
- Agent decides what to do next on its own — no manual steps needed

---

## What Is Coming Next

### 4. News Sentiment Analysis
- Fine-tuning BERT to classify financial news as positive, negative or neutral
- Sentiment score will influence the agent's investment recommendations

### 5. Vector Database and RAG
- Integrating ChromaDB as a vector database
- Enabling RAG-based retrieval so the agent can answer questions from historical stock data

### 6. Interactive Dashboard
- Building a Streamlit dashboard with Plotly visualizations
- Users will be able to query any stock and get a full AI-powered analysis

---

## Tech Stack

| Area | Tools |
|---|---|
| Data Collection | Python, yfinance, Requests |
| Data Processing | Pandas, python-dotenv |
| Agentic AI | LangGraph, LangChain, LLaMA 3.1 via Groq |
| AI & NLP | BERT, HuggingFace Transformers, RAG |
| Vector Database | ChromaDB |
| Visualization | Streamlit, Plotly |

---

## Project Structure

- `data_collector.py` — fetches 10 years of stock data for 5 companies
- `news_collector.py` — fetches live news articles
- `data_cleaner.py` — cleans and filters raw data
- `raw_data_loader.py` — loads raw Kaggle stock dataset
- `main.py` — Agentic AI pipeline with LangGraph and LLaMA 3.1

---

## Author

**Sahithi Morla**
GitHub: https://github.com/SahithiMorla123
Email: morlasaisahithi2031@gmail.com
