
from transformers import pipeline

# Load BERT sentiment analysis model
print("Loading BERT model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)
print("BERT model loaded!")


def analyze_news_sentiment(news_list: list) -> dict:
    """
    Analyze sentiment of news articles using FinBERT
    FinBERT is specially trained on financial news!
    """
    if not news_list:
        return {"sentiment": "neutral", "score": 0.0, "details": []}

    results = []
    for news in news_list:
        result = sentiment_model(news[:512])  # BERT max 512 tokens
        results.append({
            "text": news[:100],
            "sentiment": result[0]['label'].lower(),
            "score": round(result[0]['score'], 3)
        })

    # Count sentiments
    positive = sum(1 for r in results if r['sentiment'] == 'positive')
    negative = sum(1 for r in results if r['sentiment'] == 'negative')
    neutral = sum(1 for r in results if r['sentiment'] == 'neutral')

    # Overall sentiment
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


# Test it
if __name__ == "__main__":
    test_news = [
        "Apple reports record profits, stock surges 5%",
        "Apple faces lawsuit over patent infringement",
        "Apple releases new iPhone model today"
    ]

    print("\nAnalyzing news sentiment...")
    result = analyze_news_sentiment(test_news)
    print(f"Overall Sentiment: {result['overall_sentiment']}")
    print(f"Positive: {result['positive_count']}")
    print(f"Negative: {result['negative_count']}")
    print(f"Neutral: {result['neutral_count']}")
    print("\nDetails:")
    for d in result['details']:
        print(f"  {d['sentiment'].upper()} ({d['score']}) - {d['text']}")