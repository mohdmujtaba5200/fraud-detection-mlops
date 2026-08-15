"""
FastAPI Server for ML Model Predictions
Exposes trained model as REST API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Create FastAPI app
app = FastAPI(
    title="MLOps Fraud Detection API",
    description="Machine Learning API for fraud detection predictions",
    version="1.0.0"
)

# Load the trained model
MODEL_PATH = 'model.joblib'

def load_model():
    """Load trained model from disk"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_PATH)

# Load model on startup
try:
    model = load_model()
    model_loaded = True
except FileNotFoundError as e:
    model = None
    model_loaded = False
    error_message = str(e)

# Define request/response models
class PredictionRequest(BaseModel):
    """Request format for predictions"""
    features: list = [0.42, 1.37, 0.08, 0.5, 0.3, 0.9, 0.1, 0.2]
    
    class Config:
        example = {
            "features": [0.42, 1.37, 0.08, 0.5, 0.3, 0.9, 0.1, 0.2]
        }

class PredictionResponse(BaseModel):
    """Response format for predictions"""
    prediction: int
    probability: float
    is_fraud: str

# HTML interface
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }
        .container { background: white; border-radius: 10px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); max-width: 600px; padding: 40px; }
        h1 { color: #333; margin-bottom: 10px; text-align: center; }
        .subtitle { color: #666; text-align: center; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        input { width: 100%; padding: 10px; border: 2px solid #ddd; border-radius: 5px; font-size: 14px; transition: border-color 0.3s; }
        input:focus { outline: none; border-color: #667eea; }
        button { width: 100%; padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 5px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s; }
        button:hover { transform: translateY(-2px); }
        .result { margin-top: 30px; padding: 20px; background: #f5f5f5; border-radius: 5px; display: none; }
        .result.show { display: block; }
        .result-text { font-size: 18px; font-weight: 600; margin: 10px 0; }
        .fraud { color: #e74c3c; }
        .safe { color: #27ae60; }
        .info { background: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; border-radius: 3px; margin-top: 20px; }
        .info h3 { margin-bottom: 10px; color: #2c3e50; }
        .feature-input { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }
        .feature-input label { font-size: 13px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚨 Fraud Detection API</h1>
        <p class="subtitle">ML-Powered Real-time Prediction</p>
        
        <form id="predictionForm">
            <h3 style="color: #333; margin-bottom: 15px;">Enter Features (8 values)</h3>
            <div class="feature-input" id="featuresContainer"></div>
            <button type="submit">🔍 Analyze for Fraud</button>
        </form>
        
        <div class="result" id="result">
            <div class="result-text" id="resultText"></div>
            <div id="probabilityText" style="font-size: 14px; color: #666; margin-top: 10px;"></div>
        </div>
        
        <div class="info">
            <h3>ℹ️ How It Works</h3>
            <p>This API uses a trained Random Forest machine learning model to predict whether a transaction is fraudulent based on 8 features:</p>
            <ul style="margin-left: 20px; margin-top: 10px; font-size: 14px;">
                <li>Feature 1-8: Transaction characteristics</li>
                <li>Output: Fraud probability (0-1)</li>
            </ul>
        </div>
    </div>

    <script>
        // Generate input fields for 8 features
        const container = document.getElementById('featuresContainer');
        const defaultValues = [0.42, 1.37, 0.08, 0.5, 0.3, 0.9, 0.1, 0.2];
        
        for (let i = 0; i < 8; i++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.step = '0.01';
            input.min = '0';
            input.max = '1';
            input.value = defaultValues[i];
            input.placeholder = `Feature ${i + 1}`;
            input.id = `feature${i}`;
            input.name = `feature${i}`;
            
            const label = document.createElement('label');
            label.htmlFor = `feature${i}`;
            label.textContent = `Feature ${i + 1}`;
            
            const wrapper = document.createElement('div');
            wrapper.style.display = 'flex';
            wrapper.style.flexDirection = 'column';
            
            wrapper.appendChild(label);
            wrapper.appendChild(input);
            container.appendChild(wrapper);
        }

        // Handle form submission
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const features = [];
            for (let i = 0; i < 8; i++) {
                features.push(parseFloat(document.getElementById(`feature${i}`).value));
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features })
                });

                const data = await response.json();
                const resultDiv = document.getElementById('result');
                const resultText = document.getElementById('resultText');
                const probabilityText = document.getElementById('probabilityText');

                if (data.prediction === 1) {
                    resultText.innerHTML = `🚨 <span class="fraud">FRAUD DETECTED!</span>`;
                } else {
                    resultText.innerHTML = `✅ <span class="safe">Transaction is SAFE</span>`;
                }

                probabilityText.textContent = `Fraud Probability: ${(data.probability * 100).toFixed(2)}%`;
                resultDiv.classList.add('show');
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Homepage with interactive UI"""
    return HTML_CONTENT

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "message": "API is running"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make fraud prediction
    
    Request body:
    ```json
    {
      "features": [0.42, 1.37, 0.08, 0.5, 0.3, 0.9, 0.1, 0.2]
    }
    ```
    """
    
    if not model_loaded:
        raise HTTPException(
            status_code=500,
            detail=error_message
        )
    
    # Validate features
    if len(request.features) != 8:
        raise HTTPException(
            status_code=400,
            detail="Expected 8 features"
        )
    
    # Convert to numpy array
    features_array = np.array(request.features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0][1]  # Probability of fraud
    
    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability),
        is_fraud="Yes" if prediction == 1 else "No"
    )

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🚀 STARTING FASTAPI SERVER")
    print("=" * 60)
    print("📍 API running at: http://localhost:8000")
    print("📖 Docs at: http://localhost:8000/docs")
    print("🎨 Interactive UI: http://localhost:8000")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
