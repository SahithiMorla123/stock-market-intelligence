import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def load_stock_data():
    df = pd.read_csv("all_stocks_data.csv")
    return df

def clean_stock_data(df):
    print(f"Before cleaning: {df.shape}")
    df = df.dropna()
    df = df[["Date", "Open", "Close", "High", "Low", "Volume", "company"]]
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    print(f"After cleaning: {df.shape}")
    return df

def save_clean_data(df):
    df.to_csv("clean_stocks_data.csv", index=False)
    print("Clean data saved!")

df = load_stock_data()
df = clean_stock_data(df)
save_clean_data(df)