import os
import math
import requests
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent as create_agent

# LOAD ENV
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# LOAD INTERNAL MODULES
from stock_brain import analyze_news_sentiment, calculate_technical_indicators
from future_vision import predict_future
from stock_memory import query_stock_memory

# FASTAPI APP
app = FastAPI(
    title="StockVision AI",
    description="AI-powered stock analysis using LLaMA 3.3, RAG, BiLSTM and real-time data",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

llm_fast = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# REQUEST MODELS
class StockRequest(BaseModel):
    ticker: str
    period: str = "1mo"

class NewsRequest(BaseModel):
    company: str
    ticker: str = ""
    page_size: int = 5

class SentimentRequest(BaseModel):
    text: str

class AnalyzeRequest(BaseModel):
    ticker: str
    company: str
    period: str = "1mo"

class PredictRequest(BaseModel):
    ticker: str

class TechnicalRequest(BaseModel):
    ticker: str
    period: str = "3mo"

class ChatRequest(BaseModel):
    message: str
    ticker: str = ""
    company: str = ""

# AGENTIC AI TOOLS
@tool
def find_ticker(company_name: str) -> str:
    """Find the stock ticker symbol for any company name like Apple, Samsung, Oracle, BMW"""
    korean_companies = {
        "samsung": "005930.KS",
        "hyundai": "005380.KS",
        "lg": "066570.KS",
        "sk": "034730.KS",
        "lotte": "004990.KS"
    }
    for name, ticker in korean_companies.items():
        if name in company_name.lower():
            return f"Ticker for {company_name}: {ticker}"
    search = yf.Search(company_name)
    results = search.quotes
    if not results:
        return f"No ticker found for {company_name}"
    for result in results[:5]:
        ticker = result['symbol']
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d")
        if not df.empty:
            return f"Ticker for {company_name}: {ticker} ({result.get('shortname', '')})"
    return f"No live stock data found for {company_name}"

@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price for a company using ticker symbol like AAPL for Apple"""
    stock = yf.Ticker(ticker)
    df = stock.history(period="5d")
    if df.empty:
        return f"No public stock data available for {ticker}."
    latest = df.iloc[-1]
    return f"{ticker} - Latest Close: ${latest['Close']:.2f}, Volume: {latest['Volume']}"

@tool
def get_stock_history(ticker: str, period: str) -> str:
    """Get historical stock data. Period can be 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        return f"No stock history found for {ticker}"
    first = df.iloc[0]
    last  = df.iloc[-1]
    change = ((last['Close'] - first['Close']) / first['Close']) * 100
    return f"{ticker} over {period}: Start ${first['Close']:.2f} to End ${last['Close']:.2f} | Change: {change:.2f}%"

@tool
def get_company_news(company: str) -> str:
    """Get latest news articles for a company by name like Apple or Tesla"""
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    news = "\n".join([f"- {a['title']}" for a in articles])
    return news if news else "No news found"

@tool
def analyze_sentiment(news_text: str) -> str:
    """Analyze sentiment of news text using FinBERT"""
    news_list = [line.strip('- ') for line in news_text.split('\n') if line.strip()]
    result = analyze_news_sentiment(news_list)
    return f"Overall sentiment: {result['overall_sentiment']} | Positive: {result['positive_count']} | Negative: {result['negative_count']} | Neutral: {result['neutral_count']}"

@tool
def get_technical_analysis(ticker: str) -> str:
    """Get technical indicators RSI, EMA, MACD for a stock"""
    result = calculate_technical_indicators(ticker)
    if "error" in result:
        return result["error"]
    return f"RSI: {result['RSI']} ({result['RSI_signal']}) | EMA: {result['EMA_trend']} | MACD: {result['MACD_signal']}"

@tool
def predict_stock_price(ticker: str) -> str:
    """Predict stock price for next 7 days using BiLSTM + Attention model"""
    print(f"Training BiLSTM model for {ticker}... please wait...")
    result = predict_future(ticker, days=7)
    if "error" in result:
        return result["error"]
    predictions = "\n".join([f"{day}: {price}" for day, price in result['predictions'].items()])
    return f"Current Price: ${result['current_price']}\n7-Day Prediction:\n{predictions}"

@tool
def search_stock_memory(ticker: str, company: str, query: str) -> str:
    """Search ChromaDB RAG for historical stock data and news"""
    return query_stock_memory(ticker, company, query)

# AGENTIC AI SETUP
tools = [
    find_ticker, get_stock_price, get_stock_history,
    get_company_news, analyze_sentiment, get_technical_analysis,
    predict_stock_price, search_stock_memory
]

system_prompt = """You are a professional stock market analysis AI assistant.
You MUST always call ALL tools and give a detailed explanation.

For EVERY question follow these steps:
1. Call find_ticker to get the ticker symbol
2. Call get_stock_price with that ticker
3. Call get_company_news with the company name
4. Call analyze_sentiment on the news
5. Call get_technical_analysis with the ticker
6. Call search_stock_memory for historical context
7. If user asks about future, call predict_stock_price

Then give answer in this format:

CURRENT PRICE: [price]

TECHNICAL ANALYSIS:
- RSI: [value] - [what it means for investor]
- EMA: [trend] - [what it means]
- MACD: [signal] - [what it means]

NEWS SENTIMENT: [positive/negative/neutral]
- Why: [explain based on news]

HISTORICAL CONTEXT:
- [what the historical data shows]

PREDICTION CONFIDENCE: [X%]
- Based on: sentiment + technical indicators + historical patterns

RECOMMENDATION: BUY / HOLD / SELL
- Reason 1: [RSI/EMA/MACD signal]
- Reason 2: [news sentiment impact]
- Reason 3: [historical trend]

You ONLY use these tools: find_ticker, get_stock_price, get_stock_history, get_company_news, analyze_sentiment, get_technical_analysis, predict_stock_price, search_stock_memory."""

agent = create_agent(llm, tools, prompt=system_prompt)

# HELPER FUNCTIONS
def safe_float(val, default=0.0):
    """Safely convert to float - handles NaN and Infinity from Indian/Korean stocks"""
    try:
        v = float(val)
        return default if math.isnan(v) or math.isinf(v) else round(v, 2)
    except:
        return default

def safe_int(val, default=0):
    """Safely convert to int"""
    try:
        v = float(val)
        return default if math.isnan(v) or math.isinf(v) else int(v)
    except:
        return default

def get_stock_df(ticker: str, period: str = "1mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    df = df.ffill().bfill().fillna(0)
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))
    return safe_float(rsi.iloc[-1])

def calculate_macd(prices):
    ema12  = prices.ewm(span=12).mean()
    ema26  = prices.ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return safe_float(macd.iloc[-1]), safe_float(signal.iloc[-1])

def fetch_news(company: str, ticker: str = "", page_size: int = 5):
    if ticker:
        query = f'"{company}" AND (stock OR shares OR earnings OR revenue OR CEO OR quarterly)'
    else:
        query = f'"{company}" stock OR shares OR earnings'
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(query)}"
        f"&language=en&sortBy=publishedAt"
        f"&pageSize={page_size}"
        f"&apiKey={NEWS_API_KEY}"
    )
    response  = requests.get(url)
    articles  = response.json().get('articles', [])
    news_list = []
    for article in articles:
        title       = article.get('title', '') or ''
        description = article.get('description', '') or ''
        if company.lower() in title.lower() or company.lower() in description.lower():
            news_list.append({
                "title":       title,
                "description": description,
                "published":   (article.get('publishedAt', '') or '')[:10],
                "source":      article.get('source', {}).get('name', ''),
                "url":         article.get('url', '')
            })
    if not news_list:
        for article in articles:
            news_list.append({
                "title":       article.get('title', ''),
                "description": article.get('description', ''),
                "published":   (article.get('publishedAt', '') or '')[:10],
                "source":      article.get('source', {}).get('name', ''),
                "url":         article.get('url', '')
            })
    return news_list

# FASTAPI ENDPOINTS

@app.get("/")
def root():
    return {
        "message":   "StockVision AI API",
        "version":   "2.0.0",
        "endpoints": ["/price", "/news", "/sentiment", "/technical",
                      "/predict", "/analyze", "/chat", "/search"]
    }

@app.get("/search")
def search_company(q: str):
    """Search any company by name - works worldwide"""
    try:
        search = yf.Search(q, max_results=5)
        quotes = search.quotes
        if quotes:
            equity = [x for x in quotes if x.get('quoteType') == 'EQUITY']
            best   = equity[0] if equity else quotes[0]
            symbol = best['symbol']
            name   = best.get('longname') or best.get('shortname') or q
            return {"ticker": symbol, "company": name, "found": True}
        raise HTTPException(status_code=404, detail=f"Company '{q}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/price")
def get_price(request: StockRequest):
    """Get current stock price"""
    try:
        df     = get_stock_df(request.ticker, request.period)
        latest = df.iloc[-1]
        prev   = df.iloc[-2] if len(df) > 1 else latest
        price_change     = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        return {
            "ticker":           request.ticker.upper(),
            "current_price":    safe_float(latest['Close']),
            "open":             safe_float(latest['Open']),
            "high":             safe_float(latest['High']),
            "low":              safe_float(latest['Low']),
            "volume":           safe_int(latest['Volume']),
            "price_change":     safe_float(price_change),
            "price_change_pct": safe_float(price_change_pct),
            "period":           request.period,
            "total_rows":       len(df)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/news")
def get_news(request: NewsRequest):
    """Get latest news for a company"""
    try:
        news_list = fetch_news(request.company, request.ticker, request.page_size)
        return {
            "company":        request.company,
            "ticker":         request.ticker.upper() if request.ticker else "",
            "total_articles": len(news_list),
            "articles":       news_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment")
def analyze_sentiment_endpoint(request: SentimentRequest):
    """Analyze sentiment using FinBERT from stock_brain.py"""
    try:
        news_list = [request.text]
        result    = analyze_news_sentiment(news_list)
        return {
            "text":             request.text[:100],
            "sentiment":        result["overall_sentiment"].capitalize(),
            "confidence_score": safe_float(result["positive_count"] / max(len(news_list), 1)),
            "positive_count":   result["positive_count"],
            "negative_count":   result["negative_count"],
            "neutral_count":    result["neutral_count"],
            "note":             "Sentiment analyzed using FinBERT model"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/technical")
def get_technical_indicators(request: TechnicalRequest):
    """Get RSI, MACD, EMA using stock_brain.py"""
    try:
        result = calculate_technical_indicators(request.ticker)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        # Fallback for current price if stock_brain returns 0
        current_price = safe_float(result["current_price"])
        if current_price == 0.0:
            try:
                df = get_stock_df(request.ticker)
                current_price = safe_float(df.iloc[-1]['Close'])
            except:
                pass

        return {
            "ticker":        request.ticker.upper(),
            "current_price": current_price,
            "RSI":  {"value": safe_float(result["RSI"]), "signal": result["RSI_signal"]},
            "MACD": {"trend": result["MACD_signal"]},
            "EMA":  {"trend": result["EMA_trend"]}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_price(request: PredictRequest):
    """Predict next 7 days using BiLSTM + Attention from future_vision.py"""
    try:
        result = predict_future(request.ticker, days=7)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        predictions = []
        for day, price_str in result["predictions"].items():
            predictions.append({
                "day":             day,
                "predicted_price": price_str
            })
        return {
            "ticker":            request.ticker.upper(),
            "current_price":     safe_float(result["current_price"]),
            "7_day_predictions": predictions,
            "disclaimer":        "BiLSTM + Attention model prediction. Not financial advice!"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def full_analysis(request: AnalyzeRequest):
    """Full AI analysis using Agentic AI - RAG + FinBERT + BiLSTM + LLaMA"""
    try:
        query = f"Analyze {request.company} ({request.ticker}) stock - give price, technical analysis, news sentiment, historical context and BUY/HOLD/SELL recommendation"
        response = agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        ai_analysis    = response['messages'][-1].content
        recommendation = "BUY" if "BUY" in ai_analysis.upper() else "SELL" if "SELL" in ai_analysis.upper() else "HOLD"
        df       = get_stock_df(request.ticker, request.period)
        latest   = df.iloc[-1]
        week_ago = df.iloc[-5] if len(df) >= 5 else df.iloc[0]
        price_change_pct = ((latest['Close'] - week_ago['Close']) / week_ago['Close']) * 100
        return {
            "ticker":            request.ticker.upper(),
            "company":           request.company,
            "current_price":     safe_float(latest['Close']),
            "weekly_change_pct": safe_float(price_change_pct),
            "recommendation":    recommendation,
            "ai_analysis":       ai_analysis,
            "powered_by":        "LLaMA 3.3 70B + RAG + FinBERT + BiLSTM"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(request: ChatRequest):
    """Agentic AI chatbot - uses all tools automatically"""
    try:
        message = request.message
        if request.company:
            message = f"{request.message} (Company: {request.company}, Ticker: {request.ticker})"
        response = agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })
        return {
            "response":   response['messages'][-1].content,
            "powered_by": "LLaMA 3.3 70B Agentic AI + RAG + FinBERT + BiLSTM"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# RUN SERVER
if __name__ == "__main__":
    import uvicorn
    print("Starting StockVision AI Server...")
    print("Frontend: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)