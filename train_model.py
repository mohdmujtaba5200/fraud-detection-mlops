"""
Machine Learning Model Training Script
Trains a RandomForest classifier for fraud detection
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def train_model():
    """Train a RandomForest classifier and save it"""
    
    print("=" * 60)
    print("🤖 STARTING MODEL TRAINING")
    print("=" * 60)
    
    # Create synthetic dataset
    print("\n📊 Generating synthetic training data...")
    X, y = make_classification(
        n_samples=1000,      # 1000 samples
        n_features=8,        # 8 features
        n_informative=6,     # 6 are informative
        n_redundant=2,       # 2 are redundant
        n_classes=2,         # Binary classification (fraud or not)
        random_state=42
    )
    
    print(f"   ✓ Dataset shape: {X.shape}")
    print(f"   ✓ Classes: {np.unique(y)}")
    print(f"   ✓ Fraud cases: {np.sum(y)} out of {len(y)}")
    
    # Split into training and testing sets
    print("\n🔄 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    print(f"   ✓ Training set: {X_train.shape[0]} samples")
    print(f"   ✓ Test set: {X_test.shape[0]} samples")
    
    # Train RandomForest model
    print("\n🎓 Training RandomForest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,    # 100 trees
        max_depth=10,        # Max depth of tree
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1            # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    print("   ✓ Model trained successfully!")
    
    # Make predictions
    print("\n📈 Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   ✓ Accuracy:  {accuracy:.4f} (out of 1.0)")
    print(f"   ✓ Precision: {precision:.4f} (correct positive predictions)")
    print(f"   ✓ Recall:    {recall:.4f} (caught fraud cases)")
    print(f"   ✓ F1-Score:  {f1:.4f} (balanced metric)")
    
    # Save model
    print("\n💾 Saving model...")
    model_path = 'model.joblib'
    joblib.dump(model, model_path)
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"   ✓ Model saved as '{model_path}' ({file_size:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python main.py")
    print("2. Test API at: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    return model

if __name__ == "__main__":
    model = train_model()
