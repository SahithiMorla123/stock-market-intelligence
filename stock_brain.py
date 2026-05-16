import chromadb
import pandas as pd
import yfinance as yf
from datetime import date
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
stock_collection = client.get_or_create_collection(name="stock_data")
news_collection = client.get_or_create_collection(name="news_data")

# FinBERT embeddings
print("Loading finance embeddings model...")
embedder = SentenceTransformer('ProsusAI/finbert')
print("Embeddings model loaded!")

def calculate_technical_indicators(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    if df.empty:
        return {"error": f"No data found for {ticker}"}
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    latest = df.iloc[-1]
    rsi = round(latest['RSI'], 2)
    rsi_signal = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
    ema_trend = "BULLISH" if latest['EMA20'] > latest['EMA50'] else "BEARISH"
    macd_signal = "BULLISH" if latest['MACD'] > latest['Signal'] else "BEARISH"
    return {
        "ticker": ticker,
        "current_price": round(latest['Close'], 2),
        "RSI": rsi,
        "RSI_signal": rsi_signal,
        "EMA_trend": ema_trend,
        "MACD_signal": macd_signal
    }

def analyze_news_sentiment(news_list: list) -> dict:
    from transformers import pipeline
    sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    if not news_list:
        return {"overall_sentiment": "neutral", "positive_count": 0, "negative_count": 0, "neutral_count": 0}
    results = []
    for news in news_list:
        result = sentiment_model(news[:512])
        results.append(result[0]['label'].lower())
    positive = results.count('positive')
    negative = results.count('negative')
    neutral = results.count('neutral')
    overall = "positive" if positive > negative else "negative" if negative > positive else "neutral"
    return {
        "overall_sentiment": overall,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral
    }

def classify_query(query: str) -> str:
    query = query.lower()
    if any(w in query for w in ["predict", "tomorrow", "next", "future", "will"]):
        return "prediction"
    elif any(w in query for w in ["sentiment", "news", "feeling", "mood"]):
        return "sentiment"
    elif any(w in query for w in ["rsi", "macd", "technical", "indicator"]):
        return "technical"
    elif any(w in query for w in ["risk", "volatile", "safe", "danger"]):
        return "risk"
    else:
        return "general"

def fetch_and_store_company(ticker: str, company: str):
    print(f"Fetching {company} data from yfinance...")
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    if df.empty:
        print(f"No data found for {ticker}")
        return False
    df = df.reset_index()
    today = str(date.today())
    try:
        indicators = calculate_technical_indicators(ticker)
        rsi = float(indicators.get('RSI', 0))
        macd = str(indicators.get('MACD_signal', 'unknown'))
    except:
        rsi = 0.0
        macd = 'unknown'
    documents, ids, metadatas, embeddings = [], [], [], []
    for idx, row in df.iterrows():
        doc = f"Company: {ticker} Date: {str(row['Date'])[:10]} Open: {row['Open']:.2f} Close: {row['Close']:.2f} High: {row['High']:.2f} Low: {row['Low']:.2f} Volume: {row['Volume']}"
        doc_id = f"stock_{ticker}_{str(row['Date'])[:10]}"
        documents.append(doc)
        ids.append(doc_id)
        metadatas.append({
            "company": ticker,
            "date": str(row['Date'])[:10],
            "close": float(row['Close']),
            "volume": float(row['Volume']),
            "RSI": rsi,
            "MACD": macd,
            "stored_date": today
        })
        embeddings.append(embedder.encode(doc).tolist())
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        try:
            stock_collection.add(
                documents=documents[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size]
            )
        except:
            pass
    print(f"✅ {company} data stored in ChromaDB!")
    return True

def store_initial_stock_data():
    if stock_collection.count() > 0:
        print(f"✅ Stock data already stored! ({stock_collection.count()} records)")
        return
    print("Loading initial stock data...")
    df = pd.read_csv("clean_stocks_data.csv")
    print(f"Storing {len(df)} records...")
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        documents, ids, metadatas, embeddings = [], [], [], []
        for idx, row in batch.iterrows():
            doc = f"Company: {row['company']} Date: {row['Date']} Open: {row['Open']:.2f} Close: {row['Close']:.2f} High: {row['High']:.2f} Low: {row['Low']:.2f} Volume: {row['Volume']}"
            documents.append(doc)
            ids.append(f"stock_{idx}")
            metadatas.append({
                "company": str(row['company']),
                "date": str(row['Date']),
                "close": float(row['Close']),
                "volume": float(row['Volume']),
                "RSI": 0.0,
                "MACD": "unknown",
                "stored_date": str(date.today())
            })
            embeddings.append(embedder.encode(doc).tolist())
        try:
            stock_collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except:
            pass
    print("✅ Initial stock data stored!")

def store_news_with_context(company: str, ticker: str, news_list: list):
    if not news_list:
        return
    sentiment_result = analyze_news_sentiment(news_list)
    sentiment_score = round(sentiment_result['positive_count'] / max(len(news_list), 1), 2)
    try:
        indicators = calculate_technical_indicators(ticker)
        rsi = float(indicators.get('RSI', 0))
        macd = str(indicators.get('MACD_signal', 'unknown'))
    except:
        rsi = 0.0
        macd = 'unknown'
    today = str(date.today())
    for i, news in enumerate(news_list):
        try:
            embedding = embedder.encode(news).tolist()
            news_collection.add(
                documents=[news],
                ids=[f"news_{ticker}_{today}_{i}"],
                metadatas=[{
                    "company": company,
                    "ticker": ticker,
                    "sentiment": sentiment_result['overall_sentiment'],
                    "sentiment_score": sentiment_score,
                    "RSI": rsi,
                    "MACD": macd,
                    "date": today,
                    "source": "NewsAPI"
                }],
                embeddings=[embedding]
            )
        except:
            pass

def query_stock_memory(ticker: str, company: str, query: str) -> str:
    existing = stock_collection.query(
        query_texts=[f"{ticker} stock data"],
        n_results=1
    )
    if not existing['documents'][0]:
        print(f"No ChromaDB data for {ticker} — fetching from yfinance...")
        fetch_and_store_company(ticker, company)
    query_type = classify_query(query)
    try:
        stock_results = stock_collection.query(
            query_texts=[f"{ticker} {query}"],
            n_results=5
        )
        stock_docs = stock_results['documents'][0] if stock_results['documents'][0] else []
        stock_meta = stock_results['metadatas'][0] if stock_results['metadatas'][0] else []
    except:
        stock_docs = []
        stock_meta = []
    try:
        news_results = news_collection.query(
            query_texts=[f"{ticker} {query}"],
            n_results=3
        )
        news_docs = news_results['documents'][0] if news_results['documents'][0] else []
        news_meta = news_results['metadatas'][0] if news_results['metadatas'][0] else []
    except:
        news_docs = []
        news_meta = []
    response = f"Query Type: {query_type.upper()}\n\n"
    if stock_docs:
        response += "📊 Historical Data:\n"
        for doc, meta in zip(stock_docs[:3], stock_meta[:3]):
            rsi = meta.get('RSI', 'N/A')
            macd = meta.get('MACD', 'N/A')
            response += f"  {doc}\n"
            response += f"  RSI: {rsi} | MACD: {macd}\n\n"
    if news_docs:
        response += "📰 News Context:\n"
        for doc, meta in zip(news_docs[:2], news_meta[:2]):
            sentiment = meta.get('sentiment', 'unknown')
            score = float(meta.get('sentiment_score', 0))
            confidence = round(score * 100, 1)
            rsi = meta.get('RSI', 'N/A')
            response += f"  - {doc[:120]}...\n"
            response += f"    Sentiment: {sentiment.upper()} | RSI: {rsi}\n"
            response += f"    Prediction Confidence: {confidence}%\n\n"
    if not stock_docs and not news_docs:
        return f"No data found for {ticker}"
    return response

if __name__ == "__main__":
    print("Initializing Stock Memory...")
    store_initial_stock_data()
    print("\nTesting query...")
    result = query_stock_memory("NSRGY", "Nestle", "How was Nestle performing last year?")
    print(result)