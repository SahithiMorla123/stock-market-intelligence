import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent as create_agent
import yfinance as yf
import requests

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
    search = yf.Search(company_name)
    results = search.quotes
    if not results:
        return f"No ticker found for {company_name}"

    # Try each result until we find one with actual data
    for result in results[:5]:
        ticker = result['symbol']
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d")
        if not df.empty:
            return f"Ticker for {company_name}: {ticker} ({result.get('shortname', '')})"

    return f"No live stock data found for {company_name} — it may be a private company or delisted"


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


tools = [find_ticker, get_stock_price, get_stock_history, get_company_news]

system_prompt = """You are a stock market analysis AI assistant.
You MUST always call the tools to get real data. Never make up or assume data.
Follow these steps for EVERY question:
1. Call find_ticker to get the ticker symbol
2. Call get_stock_price with that ticker
3. Call get_company_news with the company name
4. Give a clear buy/sell/hold recommendation based on the real data you got
You ONLY use these tools: find_ticker, get_stock_price, get_stock_history, get_company_news.
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