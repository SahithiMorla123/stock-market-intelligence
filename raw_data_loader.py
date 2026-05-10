import pandas as pd

df = pd.read_csv("D:/C DSahithi/Downloads/Stock Market Project/archive/Stocks/aapl.us.txt")
print(df.head())
print(df.shape)
print(df.isnull().sum())
import os

path = "D:/C DSahithi/Downloads/Stock Market Project/archive/Stocks/"
all_data = []
errors = []

for filename in os.listdir(path):
    try:
        df = pd.read_csv(f"{path}{filename}")
        df["company"] = filename.replace(".us.txt", "")
        all_data.append(df)
    except Exception as e:
        errors.append(filename)

print(f"Successfully loaded: {len(all_data)} companies")
print(f"Failed to load: {len(errors)} companies")
print(f"Failed files: {errors[:10]}")