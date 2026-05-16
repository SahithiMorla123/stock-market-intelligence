import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from stock_brain import calculate_technical_indicators, analyze_news_sentiment

# Initialize ChromaDB with persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Collections
stock_collection = client.get_or_create_collection(name="stock_data")
news_collection = client.get_or_create_collection(name="news_data")

# Finance-aware embeddings
print("Loading finance embeddings model...")
embedder = SentenceTransformer('ProsusAI/finbert')
print("Embeddings model loaded!")

def store_stock_data():
    """Store stock data with technical indicators metadata"""
    # Check if already stored
    if stock_collection.count() > 0:
        print(f"✅ Stock data already stored! ({stock_collection.count()} records)")
        return
    print("Loading stock data...")
    df = pd.read_csv("clean_stocks_data.csv")
    print(f"Storing {len(df)} records in ChromaDB...")

    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        documents = []
        ids = []
        metadatas = []
        embeddings = []

        for idx, row in batch.iterrows():
            doc = f"Company: {row['company']} Date: {row['Date']} Open: {row['Open']:.2f} Close: {row['Close']:.2f} High: {row['High']:.2f} Low: {row['Low']:.2f} Volume: {row['Volume']}"
            documents.append(doc)
            ids.append(f"stock_{idx}")
            metadatas.append({
                "company": str(row['company']),
                "date": str(row['Date']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            })
            embeddings.append(embedder.encode(doc).tolist())

        stock_collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings
        )

    print("✅ Stock data stored successfully!")

def store_news_with_sentiment(company: str, ticker: str, news_list: list):
    """Store news with sentiment score, RSI, MACD metadata"""
    if not news_list:
        return

    # Get sentiment
    sentiment_result = analyze_news_sentiment(news_list)
    overall_sentiment = sentiment_result['overall_sentiment']
    sentiment_score = sentiment_result['positive_count'] / max(len(news_list), 1)

    # Get technical indicators
    try:
        indicators = calculate_technical_indicators(ticker)
        rsi = indicators.get('RSI', 0)
        macd = indicators.get('MACD_signal', 'unknown')
    except:
        rsi = 0
        macd = 'unknown'

    from datetime import date
    today = str(date.today())

    for i, news in enumerate(news_list):
        try:
            embedding = embedder.encode(news).tolist()
            news_collection.add(
                documents=[news],
                ids=[f"news_{company}_{today}_{i}"],
                metadatas=[{
                    "company": company,
                    "ticker": ticker,
                    "sentiment": overall_sentiment,
                    "sentiment_score": round(sentiment_score, 2),
                    "RSI": rsi,
                    "MACD": str(macd),
                    "date": today
                }],
                embeddings=[embedding]
            )
        except:
            pass

def classify_query(query: str) -> str:
    """Classify what type of question the user is asking"""
    query = query.lower()
    if any(w in query for w in ["predict", "tomorrow", "next", "future", "will"]):
        return "prediction"
    elif any(w in query for w in ["sentiment", "news", "feeling", "mood"]):
        return "sentiment"
    elif any(w in query for w in ["rsi", "macd", "technical", "indicator", "bollinger"]):
        return "technical"
    elif any(w in query for w in ["risk", "volatile", "safe", "danger"]):
        return "risk"
    else:
        return "general"

def query_stock_memory(company: str, query: str, ticker: str = "") -> str:
    """Smart RAG query with query classification and explanation"""
    query_type = classify_query(query)

    # Build search query based on type
    search_query = f"{company} {query}"

    # Query stock history
    try:
        stock_results = stock_collection.query(
            query_texts=[search_query],
            n_results=5
        )
        stock_docs = stock_results['documents'][0] if stock_results['documents'][0] else []
    except:
        stock_docs = []

    # Query news history
    try:
        news_results = news_collection.query(
            query_texts=[search_query],
            n_results=3
        )
        news_docs = news_results['documents'][0] if news_results['documents'][0] else []
        news_meta = news_results['metadatas'][0] if news_results['metadatas'][0] else []
    except:
        news_docs = []
        news_meta = []

    # Build explanation
    explanation = f"Query Type: {query_type.upper()}\n\n"

    if stock_docs:
        explanation += "📊 Historical Stock Data:\n"
        explanation += "\n".join(stock_docs[:3]) + "\n\n"

    if news_docs and news_meta:
        explanation += "📰 News Context:\n"
        for doc, meta in zip(news_docs[:2], news_meta[:2]):
            sentiment = meta.get('sentiment', 'unknown')
            rsi = meta.get('RSI', 'N/A')
            score = meta.get('sentiment_score', 0)
            confidence = round(score * 100, 1)
            explanation += f"- {doc[:100]}...\n"
            explanation += f"  Sentiment: {sentiment} | RSI: {rsi} | Confidence: {confidence}%\n"

    if not stock_docs and not news_docs:
        return f"No historical data found for {company}"

    return explanation

if __name__ == "__main__":
    print("Initializing Stock Memory with RAG...")
    store_stock_data()
    print("\nTesting smart query...")
    result = query_stock_memory("AAPL", "How was Apple performing in 2023?", "AAPL")
    print(result)