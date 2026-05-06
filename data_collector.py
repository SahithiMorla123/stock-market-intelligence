import yfinance as yf
import pandas as pd

def get_stock_data(company, period="1y"):
    stock = yf.Ticker(company)
    df = stock.history(period=period)
    return df

df = get_stock_data("AAPL")
print(df.head())
print(df.shape)