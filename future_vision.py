import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


# BiLSTM + Attention Model
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_days=7):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.bilstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_days)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        context = self.attention(lstm_out)
        output  = self.fc(context)
        return output


def get_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch stock data for training"""
    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)
    if df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    # Clean NaN values
    df = df.ffill().bfill().fillna(0)
    return df


def prepare_data(df: pd.DataFrame, sequence_length: int = 60):
    """Prepare data for BiLSTM training"""
    scaler      = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - 7):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i:i+7, 3])

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler


def train_model(ticker: str) -> tuple:
    """Train BiLSTM + Attention model for a specific ticker"""
    print(f"Fetching data for {ticker}...")
    df = get_stock_data(ticker)
    if df is None:
        return None, None

    print("Preparing data...")
    X, y, scaler = prepare_data(df)

    if len(X) == 0:
        return None, None

    split   = int(len(X) * 0.8)
    X_train = torch.FloatTensor(X[:split])
    y_train = torch.FloatTensor(y[:split])

    model     = BiLSTMAttention()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Training BiLSTM + Attention model...")
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_train)
        loss   = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.6f}")

    print("Training complete!")
    return model, scaler


def predict_future(ticker: str, days: int = 7) -> dict:
    """Predict future stock prices using BiLSTM + Attention"""
    df = get_stock_data(ticker)
    if df is None:
        return {"error": f"No data found for {ticker}"}

    # Get current price safely
    current_price = float(df['Close'].dropna().iloc[-1])
    if np.isnan(current_price) or current_price == 0:
        return {"error": f"Invalid price data for {ticker}"}

    model, scaler = train_model(ticker)
    if model is None:
        # Fallback to simple trend prediction if model fails
        avg_change    = float(df['Close'].pct_change().mean())
        if np.isnan(avg_change):
            avg_change = 0.001
        result = {
            "ticker":        ticker,
            "current_price": round(current_price, 2),
            "predictions":   {}
        }
        price = current_price
        for i in range(days):
            price = price * (1 + avg_change)
            change = round(((price - current_price) / current_price) * 100, 2)
            trend  = "up" if price > current_price else "down"
            result["predictions"][f"Day {i+1}"] = f"${round(price, 2)} ({trend} {abs(change)}%)"
        return result

    # Prepare last 60 days for prediction
    scaled_data = scaler.transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    last_60     = scaled_data[-60:]
    X_pred      = torch.FloatTensor(last_60).unsqueeze(0)

    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model(X_pred).numpy()[0]

    # Inverse transform predictions
    dummy          = np.zeros((7, 5))
    dummy[:, 3]    = prediction
    predicted_prices = scaler.inverse_transform(dummy)[:, 3]

    # Check for NaN - fallback to simple trend if model gives NaN
    if any(np.isnan(predicted_prices)) or any(np.isinf(predicted_prices)):
        avg_change = float(df['Close'].pct_change().mean())
        if np.isnan(avg_change):
            avg_change = 0.001
        predicted_prices = [current_price * (1 + avg_change) ** (i+1) for i in range(7)]

    result = {
        "ticker":        ticker,
        "current_price": round(current_price, 2),
        "predictions":   {}
    }

    for i in range(days):
        price  = round(float(predicted_prices[i]), 2)
        change = round(((price - current_price) / current_price) * 100, 2)
        trend  = "up" if price > current_price else "down"
        result["predictions"][f"Day {i+1}"] = f"${price} ({trend} {abs(change)}%)"

    return result


if __name__ == "__main__":
    result = predict_future("AAPL", days=7)
    print(f"\nCurrent Price: ${result['current_price']}")
    print("\n7-Day Prediction:")
    for day, price in result['predictions'].items():
        print(f"  {day}: {price}")