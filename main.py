import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent as create_agent
import yfinance as yf
import requests
from stock_brain import analyze_news_sentiment, calculate_technical_indicators
from future_vision import predict_future
from stock_memory import query_stock_memory

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

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
    last = df.iloc[-1]
    change = ((last['Close'] - first['Close']) / first['Close']) * 100
    return f"{ticker} over {period}: Start ${first['Close']:.2f} → End ${last['Close']:.2f} | Change: {change:.2f}%"

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
    """Analyze sentiment of news text using FinBERT - returns positive, negative or neutral"""
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
    """Search ChromaDB RAG for historical stock data and news for any company worldwide"""
    return query_stock_memory(ticker, company, query)

tools = [find_ticker, get_stock_price, get_stock_history, get_company_news, analyze_sentiment, get_technical_analysis, predict_stock_price, search_stock_memory]

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

📊 CURRENT PRICE: [price]

📈 TECHNICAL ANALYSIS:
- RSI: [value] → [what it means for investor]
- EMA: [trend] → [what it means]
- MACD: [signal] → [what it means]

📰 NEWS SENTIMENT: [positive/negative/neutral]
- Why: [explain based on news]

🏛️ HISTORICAL CONTEXT:
- [what the historical data shows]

🔮 PREDICTION CONFIDENCE: [X%]
- Based on: sentiment + technical indicators + historical patterns

✅ RECOMMENDATION: BUY / HOLD / SELL
- Reason 1: [RSI/EMA/MACD signal]
- Reason 2: [news sentiment impact]
- Reason 3: [historical trend]

You ONLY use these tools: find_ticker, get_stock_price, get_stock_history, get_company_news, analyze_sentiment, get_technical_analysis, predict_stock_price, search_stock_memory.
Do NOT use brave_search or any other tool."""

agent = create_agent(llm, tools, prompt=system_prompt)

print("Stock Market Agentic AI")
print("Ask me about any company!")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })
    print(f"\nAgentic AI: {response['messages'][-1].content}\n")