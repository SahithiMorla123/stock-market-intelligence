import yfinance as yf
import pandas as pd

def get_stock_data(company, period="1y"):
    stock = yf.Ticker(company)
    df = stock.history(period=period)
    return df

df = get_stock_data("AAPL")
print(df.head())
print(df.shape)
companies = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]

for company in companies:
    df = get_stock_data(company)
    df["company"] = company
    df.to_csv(f"{company}_stock_data.csv")
    print(f"Saved {company} data — {len(df)} rows")