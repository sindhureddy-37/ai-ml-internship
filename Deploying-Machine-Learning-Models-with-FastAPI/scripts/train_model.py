"""
Iris Classification Model Training Script

This script demonstrates:
1. Data loading and preprocessing
2. Model training with scikit-learn
3. Model evaluation and saving
4. Mathematical concepts behind the chosen algorithm
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

def train_iris_model():
    """
    Train a Random Forest model on the Iris dataset
    
    Random Forest Algorithm Explanation:
    - Ensemble method that combines multiple decision trees
    - Each tree is trained on a random subset of data (bootstrap sampling)
    - Final prediction is made by majority voting (classification)
    - Reduces overfitting compared to single decision trees
    - Handles feature importance naturally
    """
    
    print("Loading Iris dataset...")
    # Load the famous Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create feature names for better understanding
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")
    
    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize Random Forest Classifier
    # n_estimators: number of trees in the forest
    # random_state: for reproducible results
    # max_depth: prevents overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=3
    )
    
    print("Training the model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Feature importance (Random Forest provides this naturally)
    feature_importance = model.feature_importances_
    print("\nFeature Importance:")
    for i, importance in enumerate(feature_importance):
        print(f"{feature_names[i]}: {importance:.4f}")
    
    # Save the trained model
    model_path = "iris_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metadata for the API
    metadata = {
        "model_type": "RandomForestClassifier",
        "accuracy": float(accuracy),
        "feature_names": feature_names,
        "target_names": target_names.tolist(),
        "n_features": len(feature_names)
    }
    
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Model metadata saved to model_metadata.json")
    
    return model, metadata

if __name__ == "__main__":
    model, metadata = train_iris_model()
