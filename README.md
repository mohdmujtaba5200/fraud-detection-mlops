# 🚀 MLOps Fraud Detection API

A complete Machine Learning Operations project with a trained ML model exposed as a REST API, containerized with Docker, and orchestrated with Kubernetes.

## 📋 Project Overview

This project demonstrates a complete MLOps workflow:
- **ML Model**: Trained RandomForest classifier for fraud detection
- **API**: FastAPI web server with REST endpoints
- **Docker**: Containerized application for deployment
- **Kubernetes**: Orchestration and scaling
- **CI/CD**: Automated testing and deployment with GitHub Actions
- **Testing**: Comprehensive unit tests with pytest

---

## 🎯 Quick Start (5 minutes)

### Prerequisites
- Python 3.9+
- Docker (optional, for containerization)
- Kubernetes (optional, for orchestration)

### Setup

```bash
# 1. Clone repository
cd python.project

# 2. Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python train_model.py

# 5. Start the API server
python main.py
```

**API will be running at:** `http://localhost:8000`

---

## 📚 Project Structure

```
python.project/
├── requirements.txt              # Python dependencies
├── train_model.py                # ML model training script
├── main.py                       # FastAPI server
├── Dockerfile                    # Docker configuration
├── .gitignore                    # Git ignore rules
│
├── tests/
│   └── test_api.py              # Unit tests
│
├── k8s/
│   └── deployment.yaml          # Kubernetes manifests
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI/CD
│
├── model.joblib                 # Trained model (generated)
└── README.md                    # This file
```

---

## 🤖 ML Model Training

The model is a **RandomForest classifier** trained on synthetic fraud detection data.

### Training Script (`train_model.py`)

```bash
python train_model.py
```

**What happens:**
1. Generates 1000 synthetic transactions
2. Splits into 80% training, 20% testing
3. Trains RandomForest with 100 decision trees
4. Evaluates performance (Accuracy, Precision, Recall, F1)
5. Saves model as `model.joblib`

**Expected Output:**
```
Accuracy:  0.9500
Precision: 0.9400
Recall:    0.9600
F1-Score:  0.9500
✓ Model saved as 'model.joblib'
```

---

## 🌐 API Endpoints

### 1. **Interactive Web UI**
```
GET /
```
Browse to `http://localhost:8000` for interactive interface

### 2. **Health Check**
```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "API is running"
}
```

### 3. **Make Prediction**
```
POST /predict
```

Request body:
```json
{
  "features": [0.42, 1.37, 0.08, 0.5, 0.3, 0.9, 0.1, 0.2]
}
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.12,
  "is_fraud": "No"
}
```

---

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Test Coverage
```bash
pytest tests/ --cov=.
```

### Test Files
- `tests/test_api.py` - API endpoint tests

**Tests cover:**
- Health check endpoint
- Prediction with valid inputs
- Error handling for invalid inputs
- Different prediction scenarios

---

## 🐳 Docker Setup

### Build Docker Image
```bash
docker build -t mlops-api:latest .
```

### Run Container
```bash
docker run -p 8000:8000 mlops-api:latest
```

### Push to Registry (Optional)
```bash
docker tag mlops-api:latest myregistry/mlops-api:latest
docker push myregistry/mlops-api:latest
```

---

## ☸️ Kubernetes Deployment

### Deploy to Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
```

### Verify Deployment
```bash
# Check pods
kubectl get pods

# Check services
kubectl get svc

# View logs
kubectl logs -l app=fraud-detection-api

# Port forward (optional)
kubectl port-forward svc/fraud-detection-api-service 8000:80
```

### Scale Replicas
```bash
kubectl scale deployment fraud-detection-api --replicas=5
```

### Delete Deployment
```bash
kubectl delete -f k8s/deployment.yaml
```

---

## 🔄 CI/CD Pipeline (GitHub Actions)

Automatic testing and deployment on every push to `main` or `develop`.

### Workflow Steps
1. **Test** - Runs pytest on Python 3.9, 3.10, 3.11
2. **Build** - Creates Docker image
3. **Deploy** - Prepares for production deployment

### View Workflow Status
- Go to: `Settings > Actions`
- View detailed logs for each run

---

## 📊 Model Explanation

### What is Fraud Detection?

A binary classification task:
- **Class 0**: Transaction is legitimate (not fraud)
- **Class 1**: Transaction is fraudulent

### Features (8 input values)
Each feature represents a transaction characteristic:
- Amount
- Time
- Merchant risk score
- User location distance
- Transaction frequency
- Account age
- Payment method
- Device fingerprint

### Model Performance
- **Accuracy**: Correctly classifies transactions
- **Precision**: Of predicted fraud, how many are actual fraud
- **Recall**: Of actual fraud, how many are caught
- **F1-Score**: Balance between precision and recall

---

## 🛠 Development

### Install Dev Dependencies
```bash
pip install -r requirements.txt
pip install black flake8 mypy
```

### Code Formatting
```bash
black *.py
```

### Linting
```bash
flake8 *.py
```

### Type Checking
```bash
mypy main.py train_model.py
```

---

## 🚀 Deployment Strategies

### Local Development
```bash
python main.py
```

### Docker Container
```bash
docker run -p 8000:8000 mlops-api:latest
```

### Kubernetes Cluster
```bash
kubectl apply -f k8s/deployment.yaml
```

### Cloud Platforms
- **Render**: `pip install render-io`
- **Heroku**: `git push heroku main`
- **AWS ECS**: Push to ECR, deploy
- **GCP Cloud Run**: `gcloud run deploy`
- **Azure Container Instances**: `az container create`

---

## 📈 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Logs (Docker)
```bash
docker logs <container_id>
```

### Logs (Kubernetes)
```bash
kubectl logs -f deployment/fraud-detection-api
```

### Metrics (Optional)
Add Prometheus endpoint for metrics monitoring

---

## 🔐 Security

### Best Practices
- Use environment variables for secrets
- Validate all inputs
- Use HTTPS in production
- Implement rate limiting
- Add authentication/authorization
- Regular security audits

### Sample `.env` file
```
MODEL_PATH=model.joblib
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

---

## 📝 API Documentation

### Interactive Docs (Swagger UI)
```
http://localhost:8000/docs
```

### Alternative Docs (ReDoc)
```
http://localhost:8000/redoc
```

---

## 🤝 Contributing

1. Create a new branch
2. Make changes
3. Run tests
4. Submit pull request

---

## 📦 Dependencies

### Core
- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **scikit-learn**: ML library
- **joblib**: Model serialization

### Development
- **pytest**: Testing framework
- **httpx**: HTTP client for testing

See `requirements.txt` for all dependencies

---

## 🐛 Troubleshooting

### Model Not Found
```
Run: python train_model.py
```

### Port 8000 Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Docker Build Fails
```bash
# Clear cache
docker system prune -a

# Rebuild
docker build --no-cache -t mlops-api:latest .
```

### Kubernetes Pod Crashes
```bash
# Check logs
kubectl logs <pod_name>

# Describe pod
kubectl describe pod <pod_name>
```

---

## 📞 Support

- **Documentation**: See inline comments in code
- **Issues**: Create GitHub issue
- **Questions**: Check FAQ section

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🎓 Learning Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Scikit-Learn Guide](https://scikit-learn.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Guide](https://kubernetes.io/docs/)
- [GitHub Actions](https://github.com/features/actions)

---

**Happy deploying! 🚀**
