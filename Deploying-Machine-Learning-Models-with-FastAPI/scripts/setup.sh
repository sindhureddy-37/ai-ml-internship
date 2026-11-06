#!/bin/bash

# Setup script for the FastAPI ML deployment project

echo "Setting up FastAPI ML Deployment Project..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Train the model
echo "Training the Iris classification model..."
python scripts/train_model.py

# Start the FastAPI server
echo "Starting FastAPI server..."
echo "The API will be available at http://localhost:8000"
echo "API documentation will be available at http://localhost:8000/docs"

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
