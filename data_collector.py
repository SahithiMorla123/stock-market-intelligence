import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

COMPANIES = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
PERIOD = "10y"

def validate_company(company):
    if not isinstance(company, str) or len(company) == 0:
        raise ValueError(f"Invalid company symbol: {company}")
    return company.upper()

def get_stock_data(company, period=PERIOD):
    company = validate_company(company)
    stock = yf.Ticker(company)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for {company}")
    df["company"] = company
    return df

def collect_all_stocks():
    all_data = []
    for company in COMPANIES:
        df = get_stock_data(company)
        all_data.append(df)
        df.to_csv(f"{company}_stock_data.csv")
        print(f"Saved {company} data — {len(df)} rows")
    combined = pd.concat(all_data)
    combined.to_csv("all_stocks_data.csv")
    print(f"Combined dataset saved — {len(combined)} total rows")

collect_all_stocks()