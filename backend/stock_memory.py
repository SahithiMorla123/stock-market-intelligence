import pandas as pd
from stock_brain import (
    calculate_technical_indicators,
    analyze_news_sentiment,
    get_embedder,
    stock_collection,
    news_collection,
)

# Reuses stock_brain's single ChromaDB client/collections and lazy FinBERT
# embedder instead of opening a second copy of each.


def store_stock_data():
    """Store initial stock data from CSV into ChromaDB"""
    if stock_collection.count() > 0:
        print(f"Stock data already stored! ({stock_collection.count()} records)")
        return
    print("Loading stock data...")
    df = pd.read_csv("clean_stocks_data.csv")
    print(f"Storing {len(df)} records in ChromaDB...")

    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        documents = []
        ids       = []
        metadatas = []
        embeddings = []

        for idx, row in batch.iterrows():
            doc = f"Company: {row['company']} Date: {row['Date']} Open: {row['Open']:.2f} Close: {row['Close']:.2f} High: {row['High']:.2f} Low: {row['Low']:.2f} Volume: {row['Volume']}"
            documents.append(doc)
            ids.append(f"stock_{idx}")
            metadatas.append({
                "company": str(row['company']),
                "date":    str(row['Date']),
                "close":   float(row['Close']),
                "volume":  float(row['Volume'])
            })
            embeddings.append(get_embedder().encode(doc).tolist())

        stock_collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings
        )
    print("Stock data stored successfully!")


def auto_fetch_and_store(ticker: str, company: str):
    """Auto-fetch and store company data in ChromaDB if not already stored"""
    try:
        existing = stock_collection.query(
            query_embeddings=[get_embedder().encode(f"{ticker} stock data").tolist()],
            n_results=1,
            where={"company": ticker},
        )
        if existing['documents'][0]:
            return  # Already stored
    except:
        pass

    # Not stored — fetch and store now
    print(f"Auto-fetching {company} ({ticker}) data for ChromaDB...")
    try:
        from stock_brain import fetch_and_store_company
        fetch_and_store_company(ticker, company)
        print(f"{company} data stored in ChromaDB!")
    except Exception as e:
        print(f"Could not auto-fetch {company}: {e}")


def store_news_with_sentiment(company: str, ticker: str, news_list: list):
    """Store news with sentiment score, RSI, MACD metadata"""
    if not news_list:
        return

    sentiment_result  = analyze_news_sentiment(news_list)
    overall_sentiment = sentiment_result['overall_sentiment']
    sentiment_score   = sentiment_result['positive_count'] / max(len(news_list), 1)

    try:
        indicators = calculate_technical_indicators(ticker)
        rsi  = indicators.get('RSI', 0)
        macd = indicators.get('MACD_signal', 'unknown')
    except:
        rsi  = 0
        macd = 'unknown'

    from datetime import date
    today = str(date.today())

    for i, news in enumerate(news_list):
        try:
            embedding = get_embedder().encode(news).tolist()
            news_collection.add(
                documents=[news],
                ids=[f"news_{company}_{today}_{i}"],
                metadatas=[{
                    "company":         company,
                    "ticker":          ticker,
                    "sentiment":       overall_sentiment,
                    "sentiment_score": round(sentiment_score, 2),
                    "RSI":             rsi,
                    "MACD":            str(macd),
                    "date":            today
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
    elif any(w in query for w in ["rsi", "macd", "technical", "indicator"]):
        return "technical"
    elif any(w in query for w in ["risk", "volatile", "safe", "danger"]):
        return "risk"
    else:
        return "general"


def query_stock_memory(ticker: str, company: str, query: str) -> str:
    """Smart RAG query — auto-fetches company data if not in ChromaDB.

    Parameter order matches main.py's search_stock_memory tool, which calls
    this positionally as (ticker, company, query).
    """

    # Auto-fetch and store if company not in ChromaDB
    if ticker:
        auto_fetch_and_store(ticker, company)

    query_type   = classify_query(query)
    search_query = f"{company} {query}"
    search_embedding = get_embedder().encode(search_query).tolist()

    # Query stock history (scoped to this ticker, when known, so results
    # aren't just the nearest semantic match from a different company)
    try:
        stock_results = stock_collection.query(
            query_embeddings=[search_embedding],
            n_results=5,
            where={"company": ticker} if ticker else None,
        )
        stock_docs = stock_results['documents'][0] if stock_results['documents'][0] else []
    except:
        stock_docs = []

    # Query news history
    try:
        news_results = news_collection.query(
            query_embeddings=[search_embedding],
            n_results=3,
            where={"ticker": ticker} if ticker else None,
        )
        news_docs = news_results['documents'][0] if news_results['documents'][0] else []
        news_meta = news_results['metadatas'][0] if news_results['metadatas'][0] else []
    except:
        news_docs = []
        news_meta = []

    # Build explanation
    explanation = f"Query Type: {query_type.upper()}\n\n"

    if stock_docs:
        explanation += "Historical Stock Data:\n"
        explanation += "\n".join(stock_docs[:3]) + "\n\n"

    if news_docs and news_meta:
        explanation += "News Context:\n"
        for doc, meta in zip(news_docs[:2], news_meta[:2]):
            sentiment  = meta.get('sentiment', 'unknown')
            rsi        = meta.get('RSI', 'N/A')
            score      = meta.get('sentiment_score', 0)
            confidence = round(float(score) * 100, 1)
            explanation += f"- {doc[:100]}...\n"
            explanation += f"  Sentiment: {sentiment} | RSI: {rsi} | Confidence: {confidence}%\n"

    if not stock_docs and not news_docs:
        return f"No historical data found for {company} yet. Data is being fetched for future queries."

    return explanation


if __name__ == "__main__":
    print("Initializing Stock Memory with RAG...")
    store_stock_data()
    print("\nTesting smart query...")
    result = query_stock_memory("AAPL", "Apple", "How was Apple performing in 2023?")
    print(result)