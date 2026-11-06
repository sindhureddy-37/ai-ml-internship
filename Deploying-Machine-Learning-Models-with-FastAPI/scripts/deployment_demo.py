"""
Deployment Demo Script

This script demonstrates the complete workflow:
1. Training a model
2. Starting the API server
3. Making predictions
4. Showing real-world usage scenarios
"""

import subprocess
import time
import requests
import json
from pathlib import Path

def demo_workflow():
    """Demonstrate the complete ML deployment workflow"""
    
    print("🚀 FastAPI ML Deployment Demo")
    print("=" * 50)
    
    # Step 1: Train the model
    print("\n📊 Step 1: Training the Iris Classification Model")
    print("-" * 40)
    
    try:
        result = subprocess.run(
            ["python", "scripts/train_model.py"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Model training completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Model training failed: {e}")
        return
    
    # Step 2: Start the API server (in background)
    print("\n🌐 Step 2: Starting FastAPI Server")
    print("-" * 40)
    print("Starting server at http://localhost:8000...")
    
    # Note: In a real demo, you'd start this in a separate terminal
    print("💡 To start the server, run:")
    print("   uvicorn app.main:app --reload")
    print("\n⏳ Waiting for server to start...")
    
    # Step 3: Demo API calls
    print("\n🔮 Step 3: Making Predictions")
    print("-" * 40)
    
    # Real-world examples with different iris species
    examples = [
        {
            "name": "Iris Setosa (Small flower)",
            "data": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            "expected": "setosa"
        },
        {
            "name": "Iris Versicolor (Medium flower)",
            "data": {
                "sepal_length": 7.0,
                "sepal_width": 3.2,
                "petal_length": 4.7,
                "petal_width": 1.4
            },
            "expected": "versicolor"
        },
        {
            "name": "Iris Virginica (Large flower)",
            "data": {
                "sepal_length": 6.3,
                "sepal_width": 3.3,
                "petal_length": 6.0,
                "petal_width": 2.5
            },
            "expected": "virginica"
        }
    ]
    
    print("📝 Example API calls (run these when server is active):")
    print()
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['name']}")
        print(f"Expected: {example['expected']}")
        print("curl -X POST http://localhost:8000/predict \\")
        print("  -H 'Content-Type: application/json' \\")
        print(f"  -d '{json.dumps(example['data'])}'")
        print()
    
    # Step 4: Production considerations
    print("\n🏭 Step 4: Production Deployment")
    print("-" * 40)
    print("For production deployment, consider:")
    print("• Docker containerization")
    print("• Load balancing with multiple instances")
    print("• Model versioning and A/B testing")
    print("• Monitoring and logging")
    print("• Security (authentication, rate limiting)")
    print("• CI/CD pipeline for model updates")
    
    print("\n🐳 Docker deployment:")
    print("  docker build -t iris-api .")
    print("  docker run -p 8000:8000 iris-api")
    
    print("\n📊 Monitoring endpoints:")
    print("  Health check: GET /health")
    print("  Model info: GET /model/info")
    print("  API docs: GET /docs")

if __name__ == "__main__":
    demo_workflow()
