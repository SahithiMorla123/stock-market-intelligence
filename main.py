import os

import agent
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
    model_name="llama-3.1-8b-instant",
    temperature=0
)

@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price for a company using ticker symbol like AAPL for Apple"""
    stock = yf.Ticker(ticker)
    df = stock.history(period="5d")
    if df.empty:
        return f"No data found for {ticker}"
    latest = df.iloc[-1]
    return f"{ticker} - Latest Close: ${latest['Close']:.2f}, Volume: {latest['Volume']}"

@tool
def get_company_news(company: str) -> str:
    """Get latest news articles for a company by name like Apple or Tesla"""
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    news = "\n".join([f"- {a['title']}" for a in articles])
    return news if news else "No news found"

tools = [get_stock_price, get_company_news]

agent = create_agent(llm, tools)

response = agent.invoke({
response = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze Samsung stock - get the current price, latest news and tell me if it looks good or bad to invest today?"}]
})
})

print(response['messages'][-1].content)