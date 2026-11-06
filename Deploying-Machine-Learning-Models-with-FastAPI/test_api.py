"""
API Testing Script

This script demonstrates how to test the FastAPI endpoints
with various scenarios including edge cases and error handling
"""

import requests
import json
from typing import Dict, Any

# API base URL (adjust if running on different host/port)
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_single_prediction():
    """Test single prediction with valid data"""
    print("\nTesting single prediction...")
    
    # Sample iris measurements (this should be Setosa)
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: {result['probabilities']}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Single prediction test failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction with multiple samples"""
    print("\nTesting batch prediction...")
    
    # Multiple test samples
    test_data = [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        {
            "sepal_length": 7.0,
            "sepal_width": 3.2,
            "petal_length": 4.7,
            "petal_width": 1.4
        },
        {
            "sepal_length": 6.3,
            "sepal_width": 3.3,
            "petal_length": 6.0,
            "petal_width": 2.5
        }
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        results = response.json()
        
        for i, result in enumerate(results):
            print(f"Sample {i+1}: {result['predicted_class']} (confidence: {result['confidence']:.4f})")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Batch prediction test failed: {e}")
        return False

def test_invalid_input():
    """Test error handling with invalid input"""
    print("\nTesting invalid input handling...")
    
    # Test with negative values (should fail validation)
    invalid_data = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Error Response: {response.json()}")
        
        # Should return 422 (Validation Error)
        return response.status_code == 422
        
    except Exception as e:
        print(f"Invalid input test failed: {e}")
        return False

def test_model_info():
    """Test model information endpoint"""
    print("\nTesting model info endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        info = response.json()
        print(f"Model Type: {info.get('model_type')}")
        print(f"Accuracy: {info.get('accuracy')}")
        print(f"Features: {info.get('feature_names')}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Model info test failed: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("=" * 50)
    print("IRIS CLASSIFICATION API TESTS")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Invalid Input", test_invalid_input),
        ("Model Info", test_model_info)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        print(f"Result: {'PASS' if success else 'FAIL'}")
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

if __name__ == "__main__":
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("You can start it with: uvicorn app.main:app --reload")
    input("Press Enter to continue with tests...")
    
    run_all_tests()
