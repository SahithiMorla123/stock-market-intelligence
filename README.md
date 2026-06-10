# StockVision AI — Stock Market Intelligence Platform

StockVision AI is a full-stack AI-powered stock market analysis platform that delivers autonomous investment insights for any company worldwide. It combines Agentic AI, RAG-based retrieval, deep learning price prediction, and real-time financial data into a single intelligent system.

---

## What This Project Does

A user types any company name — Apple, Samsung, Nestle, Reliance — and the system automatically:

1. Finds the correct stock ticker for that company worldwide
2. Fetches real-time stock price, volume and historical data
3. Retrieves latest financial news about the company
4. Runs FinBERT sentiment analysis on the news
5. Calculates technical indicators — RSI, EMA, MACD
6. Retrieves historical context from ChromaDB vector database using RAG
7. Predicts next 7 days of stock prices using BiLSTM neural network
8. Generates a BUY / HOLD / SELL recommendation using LLaMA 3.3 70B
9. Asks the user to make the final investment decision — Human-in-the-Loop

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agentic AI | LangGraph + LLaMA 3.3 70B | Autonomous multi-tool reasoning and decision making |
| Language Model | LLaMA 3.3 70B | Generates investment analysis and recommendations |
| Sentiment Analysis | FinBERT (ProsusAI/finbert) | Finance-specific BERT model for news sentiment |
| Price Prediction | BiLSTM + Attention (PyTorch) | 7-day stock price forecasting |
| RAG Pipeline | ChromaDB + Sentence Transformers | Vector storage and retrieval of historical stock and news data |
| Backend | FastAPI (Python 3.11) | REST API server with 9 endpoints |
| Frontend | HTML, CSS, JavaScript | 3D glassmorphism UI inspired by Apple-style design |
| Data | yfinance | Real-time and historical stock market data |
| News | NewsAPI | Live financial news articles |
| API Testing | Postman | Manual testing of all 9 REST endpoints |
| Automated Testing | pytest | 15 automated test cases covering all endpoints |
| Version Control | Git + GitHub | Source code management |

---

## Project Structure
stock-market-intelligence/
│
├── main.py              — FastAPI server + LangGraph Agentic AI (root entry point)
├── stock_brain.py       — FinBERT sentiment analysis + RSI, EMA, MACD indicators
├── stock_memory.py      — ChromaDB RAG queries and vector storage
├── future_vision.py     — BiLSTM + Attention neural network for price prediction
├── api.py               — REST API endpoint definitions
│
├── frontend/
│   ├── index.html       — Landing page with Apple-style floating animations
│   ├── dashboard.html   — Main dashboard with stock cards and AI chatbot
│   ├── style.css        — Glassmorphism design with 3D hover effects and floating labels
│   └── script.js        — Real-time API calls, candlestick charts and animations
│
├── test_api.py          — 15 automated pytest test cases
├── chroma_db/           — ChromaDB persistent vector database
├── .env                 — Environment variables (API keys) — not pushed to GitHub
├── .gitignore           — Git ignore rules
└── requirements.txt     — Python package dependencies

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | API health check and version info |
| GET | /search | Search any company by name worldwide |
| POST | /price | Get real-time stock price and OHLCV data |
| POST | /news | Get latest financial news for a company |
| POST | /sentiment | FinBERT-powered news sentiment analysis |
| POST | /technical | RSI, MACD, EMA technical indicators |
| POST | /predict | 7-day BiLSTM price prediction |
| POST | /analyze | Full Agentic AI analysis with BUY/HOLD/SELL |
| POST | /chat | Natural language AI chatbot |

---

## Testing

### Automated Testing with pytest

15 test cases covering all endpoints:

```bash
pytest test_api.py -v
```

Tests cover valid requests, invalid tickers, missing fields, response structure and data ranges.

### Manual Testing with Postman

All 9 endpoints tested manually with multiple companies including US stocks (AAPL, TSLA, NVDA), Indian stocks (RELIANCE.NS, TCS.NS), Korean stocks (005930.KS) and European stocks (NSRGY).

---

## Frontend Design

The frontend is built with HTML, CSS and JavaScript using an Apple-style design language:

- Floating glass cards with blur background and soft borders
- 3D tilt effect on stock cards using mouse movement
- Floating labels that animate above input fields on focus — inspired by Apple product pages
- Animated background with particle network and moving stock chart lines
- Candlestick charts for 30-day price movement
- Real-time stock price cards with color-coded up and down indicators
- AI chatbot with typing indicator
- Human-in-the-Loop investment decision panel with minimize and maximize toggle
- Responsive layout with glassmorphism panels

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/SahithiMorla123/stock-market-intelligence.git
cd stock-market-intelligence
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root folder with your own API keys:
GROQ_API_KEY=your_groq_api_key
NEWS_API_KEY=your_newsapi_key

Get your free API keys here:
- Groq API key — free at console.groq.com — used to run LLaMA 3.3 70B
- NewsAPI key — free at newsapi.org — used to fetch financial news

Note: Each person who runs this project creates their own free API keys. Your keys are never shared — they stay private in your local .env file which is excluded from GitHub via .gitignore.

### 4. Start the backend

```bash
python main.py
```

Wait until you see: Application startup complete

### 5. Start the frontend

Open `frontend/index.html` with Live Server in VS Code.

### 6. Access the application

- Frontend: http://127.0.0.1:5500/frontend/index.html
- API Documentation: http://localhost:8000/docs

---

## Key Features

- Supports any company worldwide — US, India, Korea, Europe, Japan
- Agentic AI autonomously calls 8 tools without manual intervention
- RAG retrieval provides historical context from ChromaDB vector database
- BiLSTM + Attention model trains on real stock data and predicts 7-day prices
- FinBERT analyzes news sentiment with finance-specific understanding
- Human-in-the-Loop ensures final investment decisions remain with the user
- 15 automated pytest tests and full Postman manual testing for API validation

---

## Known Limitations

- Full AI analysis takes 30 to 60 seconds to respond because LLaMA 3.3 70B runs through 8 tool calls sequentially in the Agentic AI pipeline
- The /predict endpoint takes 1 to 2 minutes because BiLSTM trains a fresh neural network on every request — a pre-trained cached model would improve this significantly
- NewsAPI free tier does not always return company-specific articles — it may return general market news due to plan restrictions and limited search filtering
- Candlestick chart currently uses simulated OHLC data — real historical OHLC data from the backend needs to be connected for accurate representation
- ChromaDB RAG shows no historical context for companies not yet stored in the vector database — only the 5 initial companies have pre-stored data
- Indian and Korean stock tickers require specific exchange suffixes (.NS for NSE, .KS for Korea) which may occasionally fail to resolve automatically
- Stock price data from yfinance may have delays of 15 to 20 minutes from live market prices

---

## Disclaimer

This platform is for educational and informational purposes only. AI analysis is not financial advice. The system may not always be accurate. Always consult a certified financial advisor before making any investment decisions.

---

## Author

Sahithi Morla
GitHub: https://github.com/SahithiMorla123
Email: morlasaisahithi2031@gmail.com
