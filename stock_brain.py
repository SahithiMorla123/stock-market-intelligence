from transformers import pipeline
import yfinance as yf
import pandas as pd

# Load FinBERT sentiment analysis model
print("Loading FinBERT model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)
print("FinBERT model loaded!")


def analyze_news_sentiment(news_list: list) -> dict:
    """Analyze sentiment of news articles using FinBERT"""
    if not news_list:
        return {"overall_sentiment": "neutral", "positive_count": 0, "negative_count": 0, "neutral_count": 0,
                "details": []}

    results = []
    for news in news_list:
        result = sentiment_model(news[:512])
        results.append({
            "text": news[:100],
            "sentiment": result[0]['label'].lower(),
            "score": round(result[0]['score'], 3)
        })

    positive = sum(1 for r in results if r['sentiment'] == 'positive')
    negative = sum(1 for r in results if r['sentiment'] == 'negative')
    neutral = sum(1 for r in results if r['sentiment'] == 'neutral')

    if positive > negative:
        overall = "positive"
    elif negative > positive:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall_sentiment": overall,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "details": results
    }


def calculate_technical_indicators(ticker: str) -> dict:
    """Calculate RSI, EMA, MACD, Bollinger Bands for a stock"""
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")

    if df.empty:
        return {"error": f"No data found for {ticker}"}

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # EMA
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['BB_upper'] = df['SMA20'] + 2 * df['Close'].rolling(20).std()
    df['BB_lower'] = df['SMA20'] - 2 * df['Close'].rolling(20).std()

    latest = df.iloc[-1]
    rsi = round(latest['RSI'], 2)

    # RSI interpretation
    if rsi < 30:
        rsi_signal = "OVERSOLD - Good buying opportunity!"
    elif rsi > 70:
        rsi_signal = "OVERBOUGHT - Consider selling!"
    else:
        rsi_signal = "NEUTRAL - Normal range"

    # EMA trend
    if latest['EMA20'] > latest['EMA50']:
        ema_trend = "BULLISH - Short term trend above long term"
    else:
        ema_trend = "BEARISH - Short term trend below long term"

    # MACD signal
    if latest['MACD'] > latest['Signal']:
        macd_signal = "BULLISH - Momentum is positive"
    else:
        macd_signal = "BEARISH - Momentum is negative"

    # Bollinger Bands
    if latest['Close'] < latest['BB_lower']:
        bb_signal = "Price below lower band - Potential BUY signal"
    elif latest['Close'] > latest['BB_upper']:
        bb_signal = "Price above upper band - Potential SELL signal"
    else:
        bb_signal = "Price within normal range"

    return {
        "ticker": ticker,
        "current_price": round(latest['Close'], 2),
        "RSI": rsi,
        "RSI_signal": rsi_signal,
        "EMA_trend": ema_trend,
        "MACD_signal": macd_signal,
        "BB_signal": bb_signal
    }


# Test
if __name__ == "__main__":
    print("\n--- Testing Sentiment ---")
    test_news = [
        "Apple reports record profits, stock surges 5%",
        "Apple faces lawsuit over patent infringement",
        "Apple releases new iPhone model today"
    ]
    result = analyze_news_sentiment(test_news)
    print(f"Overall Sentiment: {result['overall_sentiment']}")

    print("\n--- Testing Technical Indicators ---")
    indicators = calculate_technical_indicators("AAPL")
    for key, value in indicators.items():
        print(f"{key}: {value}")