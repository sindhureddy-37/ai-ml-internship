"""
FastAPI Application for Iris Classification

This FastAPI application demonstrates:
1. Loading a pre-trained ML model
2. Creating RESTful API endpoints
3. Input validation with Pydantic models
4. Error handling and logging
5. API documentation with OpenAPI/Swagger
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import json
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Iris Classification API",
    description="A machine learning API for classifying Iris flowers based on sepal and petal measurements",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc documentation
)

# Global variables for model and metadata
model = None
metadata = None

# Pydantic models for request/response validation
class IrisFeatures(BaseModel):
    """
    Input model for Iris classification
    
    Validates that all measurements are positive numbers
    within reasonable ranges for iris flowers
    """
    sepal_length: float = Field(
        ..., 
        ge=0.1, 
        le=10.0, 
        description="Sepal length in centimeters"
    )
    sepal_width: float = Field(
        ..., 
        ge=0.1, 
        le=10.0, 
        description="Sepal width in centimeters"
    )
    petal_length: float = Field(
        ..., 
        ge=0.1, 
        le=10.0, 
        description="Petal length in centimeters"
    )
    petal_width: float = Field(
        ..., 
        ge=0.1, 
        le=10.0, 
        description="Petal width in centimeters"
    )
    
    @validator('*', pre=True)
    def validate_numeric(cls, v):
        """Ensure all inputs are valid numbers"""
        if not isinstance(v, (int, float)):
            raise ValueError('All measurements must be numeric')
        return float(v)

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class: str = Field(..., description="Predicted iris species")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    input_features: Dict[str, float] = Field(..., description="Input features used for prediction")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]

@app.on_event("startup")
async def load_model():
    """
    Load the trained model and metadata on application startup
    
    This is called once when the FastAPI server starts
    """
    global model, metadata
    
    try:
        # Load the trained model
        model_path = Path("iris_model.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        # Load metadata
        metadata_path = Path("model_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info("Model metadata loaded successfully")
        else:
            logger.warning("Model metadata not found, using defaults")
            metadata = {
                "model_type": "Unknown",
                "accuracy": 0.0,
                "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                "target_names": ["setosa", "versicolor", "virginica"],
                "n_features": 4
            }
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Iris Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns the current status of the API and model
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_info=metadata or {}
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_iris(features: IrisFeatures):
    """
    Predict iris species based on flower measurements
    
    This endpoint:
    1. Validates input data using Pydantic
    2. Converts input to numpy array
    3. Makes prediction using the loaded model
    4. Returns prediction with confidence scores
    
    Args:
        features: IrisFeatures object with sepal/petal measurements
        
    Returns:
        PredictionResponse with prediction, confidence, and probabilities
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert input to numpy array in the correct order
        input_array = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_array)[0]
        
        # Get class names
        target_names = metadata.get("target_names", ["setosa", "versicolor", "virginica"])
        predicted_class = target_names[prediction]
        
        # Calculate confidence (max probability)
        confidence = float(np.max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            target_names[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Create input features dictionary
        input_dict = {
            "sepal_length": features.sepal_length,
            "sepal_width": features.sepal_width,
            "petal_length": features.petal_length,
            "petal_width": features.petal_width
        }
        
        logger.info(f"Prediction made: {predicted_class} (confidence: {confidence:.4f})")
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            input_features=input_dict
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(features_list: List[IrisFeatures]):
    """
    Batch prediction endpoint for multiple samples
    
    Args:
        features_list: List of IrisFeatures objects
        
    Returns:
        List of PredictionResponse objects
    """
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    if len(features_list) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 100 samples allowed."
        )
    
    try:
        results = []
        for features in features_list:
            # Reuse the single prediction logic
            prediction_result = await predict_iris(features)
            results.append(prediction_result)
        
        logger.info(f"Batch prediction completed for {len(features_list)} samples")
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """
    Get information about the loaded model
    
    Returns model metadata including accuracy, feature names, etc.
    """
    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model metadata not available"
        )
    
    return metadata

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return HTTPException(
        status_code=400,
        detail=f"Invalid input: {str(exc)}"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
