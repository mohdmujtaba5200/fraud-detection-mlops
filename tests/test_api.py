"""
Unit tests for Fraud Detection API
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root():
    """Test root endpoint returns HTML"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Fraud Detection" in response.text

def test_predict_valid():
    """Test prediction with valid input"""
    payload = {
        "features": [0.42, 1.37, 0.08, 0.5, 0.3, 0.9, 0.1, 0.2]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "is_fraud" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1

def test_predict_invalid_features_count():
    """Test prediction with wrong number of features"""
    payload = {
        "features": [0.5, 0.5, 0.5]  # Only 3 features instead of 8
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400

def test_predict_different_values():
    """Test prediction with different feature values"""
    payload = {
        "features": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["probability"] >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
