import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

st.set_page_config(
    page_title="Stock Intelligence AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; color: #e0e0e0; }
    .main-title {
        font-family: monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #00ff88;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        margin-top: 0;
    }
    .metric-card {
        background: #111118;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-family: monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00ff88;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
    }
    .news-card {
        background: #111118;
        border-left: 3px solid #00a8ff;
        border-radius: 0 8px 8px 0;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00ff88, #00a8ff);
        color: #000;
        font-weight: 700;
        border: none;
        border-radius: 8px;
    }
    div[data-testid="stSidebar"] {
        background: #080810;
        border-right: 1px solid #2a2a3a;
    }
</style>
""", unsafe_allow_html=True)

COMPANY_MAP = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Meta": "META",
    "Nvidia": "NVDA",
    "Netflix": "NFLX"
}

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

@st.cache_data(ttl=600)
def fetch_news(company, api_key):
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&pageSize=10&apiKey={api_key}"
    try:
        response = requests.get(url)
        return response.json().get('articles', [])
    except:
        return []

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def simple_sentiment(text):
    positive_words = ['surge', 'gain', 'profit', 'growth', 'record', 'beat', 'strong', 'rise', 'up', 'high', 'launch', 'success']
    negative_words = ['fall', 'drop', 'loss', 'decline', 'miss', 'weak', 'down', 'low', 'cut', 'layoff', 'lawsuit', 'crash']
    text_lower = text.lower()
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    if pos > neg: return "Positive"
    elif neg > pos: return "Negative"
    else: return "Neutral"

def get_ai_recommendation(company, ticker, stock_data, news_articles, llm):
    if stock_data.empty:
        return "Unable to analyze — no stock data available."
    latest = stock_data.iloc[-1]
    week_ago = stock_data.iloc[-5] if len(stock_data) >= 5 else stock_data.iloc[0]
    price_change = ((latest['Close'] - week_ago['Close']) / week_ago['Close']) * 100
    news_summary = "\n".join([f"- {a['title']}" for a in news_articles[:5]])
    messages = [
        SystemMessage(content="You are a senior financial analyst AI. Give a clear BUY, HOLD, or SELL recommendation. Always cite news sources. Keep it concise."),
        HumanMessage(content=f"""
Company: {company} ({ticker})
Current Price: ${latest['Close']:.2f}
Weekly Change: {price_change:.2f}%

Recent News:
{news_summary}

Provide a BUY/HOLD/SELL recommendation with reasoning.
""")
    ]
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 SEARCH")
    company_name = st.selectbox("Select Company", list(COMPANY_MAP.keys()))
    custom_ticker = st.text_input("Or enter custom ticker", "")
    ticker = custom_ticker.upper() if custom_ticker else COMPANY_MAP[company_name]

    st.markdown("### 📅 TIME PERIOD")
    period_option = st.selectbox("Quick Select", ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"])

    end_date = datetime.now()
    if period_option == "1 Week": start_date = end_date - timedelta(days=7)
    elif period_option == "1 Month": start_date = end_date - timedelta(days=30)
    elif period_option == "3 Months": start_date = end_date - timedelta(days=90)
    elif period_option == "6 Months": start_date = end_date - timedelta(days=180)
    elif period_option == "1 Year": start_date = end_date - timedelta(days=365)
    elif period_option == "2 Years": start_date = end_date - timedelta(days=730)
    else: start_date = end_date - timedelta(days=1825)

    st.markdown("### ⚙️ INDICATORS")
    show_ema = st.checkbox("EMA (20/50)", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_volume = st.checkbox("Volume", value=True)
    show_bollinger = st.checkbox("Bollinger Bands", value=False)

    analyze_btn = st.button("🚀 ANALYZE", use_container_width=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📈 STOCK INTELLIGENCE AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by LLaMA 3.1 · RAG · Real-time Data</p>', unsafe_allow_html=True)
st.markdown("---")

with st.spinner(f"Fetching data for {ticker}..."):
    stock_data = fetch_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    news_articles = fetch_news(company_name, os.getenv("NEWS_API_KEY", ""))

if stock_data.empty:
    st.error(f"No data found for {ticker}!")
else:
    latest = stock_data.iloc[-1]
    prev = stock_data.iloc[-2] if len(stock_data) > 1 else latest
    price_change = latest['Close'] - prev['Close']
    price_change_pct = (price_change / prev['Close']) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${latest["Close"]:.2f}</div><div class="metric-label">Price</div></div>', unsafe_allow_html=True)
    with col2:
        color = "#00ff88" if price_change >= 0 else "#ff4444"
        arrow = "▲" if price_change >= 0 else "▼"
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">{arrow} {price_change_pct:.2f}%</div><div class="metric-label">Change</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${latest["High"]:.2f}</div><div class="metric-label">High</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${latest["Low"]:.2f}</div><div class="metric-label">Low</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{latest["Volume"]/1e6:.1f}M</div><div class="metric-label">Volume</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    chart_col, news_col = st.columns([3, 1])

    with chart_col:
        rows = 1
        row_heights = [0.7]
        if show_rsi:
            rows += 1
            row_heights.append(0.15)
        if show_volume:
            rows += 1
            row_heights.append(0.15)

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#003322',
            decreasing_fillcolor='#330000'
        ), row=1, col=1)

        if show_ema:
            ema20 = stock_data['Close'].ewm(span=20).mean()
            ema50 = stock_data['Close'].ewm(span=50).mean()
            fig.add_trace(go.Scatter(x=stock_data.index, y=ema20, name="EMA 20", line=dict(color='#00a8ff', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=ema50, name="EMA 50", line=dict(color='#ffaa00', width=1.5)), row=1, col=1)

        if show_bollinger:
            sma20 = stock_data['Close'].rolling(20).mean()
            std20 = stock_data['Close'].rolling(20).std()
            fig.add_trace(go.Scatter(x=stock_data.index, y=sma20 + std20*2, name="BB Upper", line=dict(color='#aa44ff', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=sma20 - std20*2, name="BB Lower", line=dict(color='#aa44ff', width=1, dash='dash')), row=1, col=1)

        current_row = 2
        if show_rsi:
            rsi = calculate_rsi(stock_data['Close'])
            fig.add_trace(go.Scatter(x=stock_data.index, y=rsi, name="RSI", line=dict(color='#ff88aa', width=1.5)), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#ff4444", opacity=0.5, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", opacity=0.5, row=current_row, col=1)
            current_row += 1

        if show_volume:
            colors = ['#00ff88' if stock_data['Close'].iloc[i] >= stock_data['Open'].iloc[i] else '#ff4444' for i in range(len(stock_data))]
            fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name="Volume", marker_color=colors, opacity=0.7), row=current_row, col=1)

        fig.update_layout(
            paper_bgcolor='#0a0a0f', plot_bgcolor='#0d0d15',
            font=dict(color='#888'),
            xaxis_rangeslider_visible=False,
            height=550,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(gridcolor='#1a1a2a'),
            yaxis=dict(gridcolor='#1a1a2a')
        )
        st.plotly_chart(fig, use_container_width=True)

    with news_col:
        st.markdown("### 📰 NEWS")
        for article in news_articles[:6]:
            title = article.get('title', '')
            url = article.get('url', '#')
            published = article.get('publishedAt', '')[:10]
            sentiment = simple_sentiment(title)
            color = "#00ff88" if sentiment == "Positive" else "#ff4444" if sentiment == "Negative" else "#ffaa00"
            st.markdown(f"""
            <div class="news-card">
                <div style="font-size:0.75rem; color:{color}; margin-bottom:5px">{published} · {sentiment}</div>
                <a href="{url}" target="_blank" style="color:#00a8ff; font-size:0.85rem; text-decoration:none">{title[:100]}...</a>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 AI RECOMMENDATION")

    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        with st.spinner("AI analyzing..."):
            llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant", temperature=0)
            ai_analysis = get_ai_recommendation(company_name, ticker, stock_data, news_articles, llm)

        if "BUY" in ai_analysis.upper(): bg = "#003322"; border = "#00ff88"; icon = "🟢 BUY"
        elif "SELL" in ai_analysis.upper(): bg = "#330000"; border = "#ff4444"; icon = "🔴 SELL"
        else: bg = "#332200"; border = "#ffaa00"; icon = "🟡 HOLD"

        st.markdown(f"""
        <div style="background:{bg}; border:1px solid {border}; border-radius:12px; padding:20px">
            <div style="font-size:1.2rem; font-weight:700; margin-bottom:10px">{icon}</div>
            <div style="font-size:0.9rem; line-height:1.6; color:#ddd">{ai_analysis}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 👤 HUMAN-IN-THE-LOOP")
        if 'decision' not in st.session_state:
            st.session_state.decision = None

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✅ APPROVE", use_container_width=True):
                st.session_state.decision = "approved"
        with c2:
            if st.button("✏️ EDIT", use_container_width=True):
                st.session_state.decision = "editing"
        with c3:
            if st.button("❌ REJECT", use_container_width=True):
                st.session_state.decision = "rejected"

        if st.session_state.decision == "approved":
            st.success("✅ Recommendation approved!")
        elif st.session_state.decision == "rejected":
            st.error("❌ Recommendation rejected!")
        elif st.session_state.decision == "editing":
            edited = st.text_area("Edit analysis:", value=ai_analysis, height=150)
            if st.button("💾 Save"):
                st.session_state.decision = "approved"
                st.success("✅ Saved!")
    else:
        st.warning("Add GROQ_API_KEY to .env to enable AI!")

    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#444; font-size:0.75rem">STOCK INTELLIGENCE AI · Not financial advice</div>', unsafe_allow_html=True)