import requests
import pandas as pd

API_KEY = "67faaf07c85645cb9a11f22261f82187"

def get_news(company_name):
    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = data['articles']
    news_list = []
    for article in articles[:10]:
        news_list.append({
            "company": company_name,
            "title": article['title'],
            "description": article['description'],
            "published": article['publishedAt'],
            "url": article['url']
        })
    df = pd.DataFrame(news_list)
    return df

df = get_news("Apple")
print(df.head())
print(f"Total articles: {len(df)}")
df.to_csv("apple_news.csv", index=False)
print("News saved!")