import os
import requests
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(
    title="Stock Market Intelligence API",
    description="AI-powered stock analysis using LLaMA 3.1, RAG and real-time data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ── REQUEST MODELS ─────────────────────────────────────────────────────────
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

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────
def get_stock_df(ticker: str, period: str = "1mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calculate_macd(prices):
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return round(macd.iloc[-1], 4), round(signal.iloc[-1], 4)

def simple_sentiment(text: str):
    positive_words = ['surge', 'gain', 'profit', 'growth', 'record', 'beat', 'strong',
                      'rise', 'up', 'high', 'launch', 'success', 'bullish', 'buy',
                      'rally', 'outperform', 'upgrade', 'boost', 'expand', 'revenue']
    negative_words = ['fall', 'drop', 'loss', 'decline', 'miss', 'weak', 'down', 'low',
                      'cut', 'layoff', 'lawsuit', 'crash', 'bearish', 'sell', 'downgrade',
                      'risk', 'concern', 'warn', 'debt', 'short']
    text_lower = text.lower()
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    if pos > neg:
        return {"sentiment": "Positive", "score": round(pos / (pos + neg + 1), 2)}
    elif neg > pos:
        return {"sentiment": "Negative", "score": round(neg / (pos + neg + 1), 2)}
    else:
        return {"sentiment": "Neutral", "score": 0.5}

def fetch_news(company: str, ticker: str = "", page_size: int = 5):
    if ticker:
        query = f'"{company}" AND (stock OR shares OR earnings OR revenue OR CEO OR quarterly)'
    else:
        query = f'"{company}" stock OR shares OR earnings'

    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(query)}"
        f"&language=en"
        f"&sortBy=publishedAt"
        f"&pageSize={page_size}"
        f"&apiKey={NEWS_API_KEY}"
    )
    response = requests.get(url)
    articles = response.json().get('articles', [])

    news_list = []
    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        if company.lower() in title.lower() or company.lower() in description.lower():
            news_list.append({
                "title": title,
                "description": description,
                "published": (article.get('publishedAt', '') or '')[:10],
                "source": article.get('source', {}).get('name', ''),
                "url": article.get('url', '')
            })

    if not news_list:
        for article in articles:
            news_list.append({
                "title": article.get('title', ''),
                "description": article.get('description', ''),
                "published": (article.get('publishedAt', '') or '')[:10],
                "source": article.get('source', {}).get('name', ''),
                "url": article.get('url', '')
            })

    return news_list

# ── ENDPOINTS ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Stock Market Intelligence API",
        "version": "1.0.0",
        "endpoints": ["/price", "/news", "/sentiment", "/technical", "/predict", "/analyze", "/search"]
    }

@app.get("/search")
def search_company(q: str):
    """Search for any company ticker by name — works for any company worldwide"""
    try:
        search = yf.Search(q, max_results=5)
        quotes = search.quotes
        if quotes:
            # Filter for equity type results first
            equity = [x for x in quotes if x.get('quoteType') == 'EQUITY']
            best = equity[0] if equity else quotes[0]
            symbol  = best['symbol']
            name    = best.get('longname') or best.get('shortname') or q
            return {"ticker": symbol, "company": name, "found": True}
        raise HTTPException(status_code=404, detail=f"Company '{q}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/price")
def get_price(request: StockRequest):
    try:
        df = get_stock_df(request.ticker, request.period)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        return {
            "ticker": request.ticker.upper(),
            "current_price": round(float(latest['Close']), 2),
            "open": round(float(latest['Open']), 2),
            "high": round(float(latest['High']), 2),
            "low": round(float(latest['Low']), 2),
            "volume": int(latest['Volume']),
            "price_change": round(float(price_change), 2),
            "price_change_pct": round(float(price_change_pct), 2),
            "period": request.period,
            "total_rows": len(df)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/news")
def get_news(request: NewsRequest):
    try:
        news_list = fetch_news(request.company, request.ticker, request.page_size)
        return {
            "company": request.company,
            "ticker": request.ticker.upper() if request.ticker else "",
            "total_articles": len(news_list),
            "articles": news_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment")
def analyze_sentiment(request: SentimentRequest):
    result = simple_sentiment(request.text)
    return {
        "text": request.text[:100],
        "sentiment": result["sentiment"],
        "confidence_score": result["score"],
        "note": "Sentiment scored using financial keyword analysis"
    }

@app.post("/technical")
def get_technical_indicators(request: TechnicalRequest):
    try:
        df = get_stock_df(request.ticker, request.period)
        prices = df['Close']
        rsi = calculate_rsi(prices)
        macd, signal = calculate_macd(prices)
        ema20 = round(float(prices.ewm(span=20).mean().iloc[-1]), 2)
        ema50 = round(float(prices.ewm(span=50).mean().iloc[-1]), 2)
        current_price = round(float(prices.iloc[-1]), 2)

        if rsi > 70:
            rsi_signal = "Overbought — potential sell signal"
        elif rsi < 30:
            rsi_signal = "Oversold — potential buy signal"
        else:
            rsi_signal = "Neutral"

        macd_signal = "Bullish" if macd > signal else "Bearish"
        ema_signal  = "Bullish" if ema20 > ema50 else "Bearish"

        return {
            "ticker": request.ticker.upper(),
            "current_price": current_price,
            "RSI":  {"value": rsi,  "signal": rsi_signal},
            "MACD": {"macd": macd,  "signal_line": signal, "trend": macd_signal},
            "EMA":  {"ema_20": ema20, "ema_50": ema50, "trend": ema_signal}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_price(request: PredictRequest):
    try:
        df = get_stock_df(request.ticker, "3mo")
        prices = df['Close']
        ema20 = prices.ewm(span=20).mean().iloc[-1]
        ema50 = prices.ewm(span=50).mean().iloc[-1]
        rsi   = calculate_rsi(prices)
        current_price     = float(prices.iloc[-1])
        avg_daily_change  = float(prices.pct_change().mean())

        predictions = []
        price = current_price
        for i in range(1, 8):
            price = price * (1 + avg_daily_change)
            predictions.append({"day": f"Day {i}", "predicted_price": round(price, 2)})

        if ema20 > ema50 and rsi < 70:
            trend = "Bullish — upward trend expected"
        elif ema20 < ema50 and rsi > 30:
            trend = "Bearish — downward trend expected"
        else:
            trend = "Neutral — sideways movement expected"

        return {
            "ticker": request.ticker.upper(),
            "current_price": round(current_price, 2),
            "trend": trend,
            "7_day_predictions": predictions,
            "disclaimer": "This is a simple statistical prediction, not financial advice!"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def full_analysis(request: AnalyzeRequest):
    try:
        df = get_stock_df(request.ticker, request.period)
        latest   = df.iloc[-1]
        week_ago = df.iloc[-5] if len(df) >= 5 else df.iloc[0]
        price_change_pct = ((latest['Close'] - week_ago['Close']) / week_ago['Close']) * 100

        articles_raw = fetch_news(request.company, request.ticker, 5)
        news_summary = "\n".join([f"- {a['title']}" for a in articles_raw[:5]])

        rsi  = calculate_rsi(df['Close'])
        macd, signal = calculate_macd(df['Close'])

        messages = [
            SystemMessage(content="""You are a senior financial analyst AI.
Analyze the stock data and news provided.
Give a clear BUY, HOLD, or SELL recommendation.
Always cite your sources from the news provided.
Keep your analysis concise but insightful."""),
            HumanMessage(content=f"""
Company: {request.company} ({request.ticker})
Current Price: ${latest['Close']:.2f}
Weekly Change: {price_change_pct:.2f}%
RSI: {rsi}
MACD: {macd}

Recent News:
{news_summary}

Provide a BUY/HOLD/SELL recommendation with reasoning.
""")
        ]

        ai_response    = llm.invoke(messages)
        recommendation = "BUY" if "BUY" in ai_response.content.upper() else "SELL" if "SELL" in ai_response.content.upper() else "HOLD"

        return {
            "ticker": request.ticker.upper(),
            "company": request.company,
            "current_price": round(float(latest['Close']), 2),
            "weekly_change_pct": round(float(price_change_pct), 2),
            "technical_indicators": {"RSI": rsi, "MACD": macd, "signal_line": signal},
            "recommendation": recommendation,
            "ai_analysis": ai_response.content,
            "news_used": [a['title'] for a in articles_raw[:3]]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))