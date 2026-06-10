import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ── ROOT ──────────────────────────────────────────────────────────────────────

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data

# ── PRICE ─────────────────────────────────────────────────────────────────────

def test_price_valid_ticker():
    response = client.post("/price", json={"ticker": "AAPL"})
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["current_price"] > 0

def test_price_invalid_ticker():
    response = client.post("/price", json={"ticker": "INVALIDXYZ123"})
    assert response.status_code == 404

def test_price_missing_ticker():
    response = client.post("/price", json={})
    assert response.status_code == 422

# ── SENTIMENT ─────────────────────────────────────────────────────────────────

def test_sentiment_positive():
    response = client.post("/sentiment", json={"text": "Apple stock surged to record high with strong profit growth"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "Positive"

def test_sentiment_negative():
    response = client.post("/sentiment", json={"text": "Stock crash and massive loss with layoffs and decline"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "Negative"

def test_sentiment_missing_text():
    response = client.post("/sentiment", json={})
    assert response.status_code == 422

# ── TECHNICAL ─────────────────────────────────────────────────────────────────

def test_technical_valid_ticker():
    response = client.post("/technical", json={"ticker": "AAPL"})
    assert response.status_code == 200
    data = response.json()
    assert "RSI" in data
    assert "MACD" in data
    assert "EMA" in data

def test_technical_rsi_range():
    response = client.post("/technical", json={"ticker": "TSLA"})
    assert response.status_code == 200
    rsi = response.json()["RSI"]["value"]
    assert 0 <= rsi <= 100

def test_technical_missing_ticker():
    response = client.post("/technical", json={})
    assert response.status_code == 422

# ── PREDICT ───────────────────────────────────────────────────────────────────

def test_predict_valid_ticker():
    response = client.post("/predict", json={"ticker": "AAPL"})
    assert response.status_code == 200
    assert len(response.json()["7_day_predictions"]) == 7

def test_predict_has_disclaimer():
    response = client.post("/predict", json={"ticker": "AMZN"})
    assert response.status_code == 200
    assert "disclaimer" in response.json()

def test_predict_missing_ticker():
    response = client.post("/predict", json={})
    assert response.status_code == 422

# ── ANALYZE ───────────────────────────────────────────────────────────────────

def test_analyze_valid_request():
    response = client.post("/analyze", json={"ticker": "AAPL", "company": "Apple"})
    assert response.status_code == 200
    assert response.json()["recommendation"] in ["BUY", "HOLD", "SELL"]

def test_analyze_missing_company():
    response = client.post("/analyze", json={"ticker": "AAPL"})
    assert response.status_code == 422