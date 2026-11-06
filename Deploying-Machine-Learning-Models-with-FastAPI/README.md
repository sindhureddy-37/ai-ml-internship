# Iris Classification API with FastAPI

A production-ready machine learning API for classifying Iris flower species using FastAPI and scikit-learn. This project demonstrates best practices for deploying ML models as RESTful web services.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides a complete solution for deploying a machine learning model that classifies Iris flowers into three species:
- **Iris Setosa** - Small flowers with short petals
- **Iris Versicolor** - Medium-sized flowers
- **Iris Virginica** - Large flowers with long petals

The API accepts flower measurements (sepal length/width, petal length/width) and returns the predicted species with confidence scores.

## ✨ Features

- 🚀 **Fast & Scalable**: Built with FastAPI for high-performance async operations
- 📊 **ML-Powered**: Random Forest classifier with 95%+ accuracy
- 🔍 **Input Validation**: Automatic request validation with Pydantic models
- 📚 **Auto Documentation**: Interactive API docs with Swagger UI
- 🐳 **Docker Ready**: Containerized for easy deployment
- 🧪 **Comprehensive Testing**: Full test suite with edge cases
- 📈 **Production Ready**: Error handling, logging, and health checks
- 🔄 **Batch Processing**: Support for multiple predictions in one request

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd iris-classification-api
   \`\`\`

2. **Create virtual environment**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

3. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Train the model**
   \`\`\`bash
   python scripts/train_model.py
   \`\`\`

5. **Start the API server**
   \`\`\`bash
   uvicorn app.main:app --reload
   \`\`\`

6. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Quick Test

\`\`\`bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
\`\`\`

Expected response:
\`\`\`json
{
  "predicted_class": "setosa",
  "confidence": 0.98,
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.02,
    "virginica": 0.00
  },
  "input_features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
\`\`\`

## 📖 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint with API info |
| GET | `/health` | Health check and model status |
| GET | `/model/info` | Model metadata and performance |
| POST | `/predict` | Single flower classification |
| POST | `/predict/batch` | Batch classification (up to 100 samples) |
| GET | `/docs` | Interactive API documentation |

### Request Format

**Single Prediction:**
\`\`\`json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
\`\`\`

**Batch Prediction:**
\`\`\`json
[
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
  }
]
\`\`\`

### Response Format

\`\`\`json
{
  "predicted_class": "setosa",
  "confidence": 0.98,
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.02,
    "virginica": 0.00
  },
  "input_features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
\`\`\`

### Input Validation

All measurements must be:
- Numeric values (int or float)
- Between 0.1 and 10.0 centimeters
- Non-negative

Invalid inputs return HTTP 422 with detailed error messages.

## 📁 Project Structure

\`\`\`
iris-classification-api/
├── app/
│   └── main.py              # FastAPI application
├── scripts/
│   ├── train_model.py       # Model training script
│   ├── setup.sh            # Project setup script
│   └── deployment_demo.py   # Demo workflow
├── test_api.py             # API testing suite
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Multi-service deployment
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── iris_model.joblib      # Trained model (generated)
└── model_metadata.json    # Model info (generated)
\`\`\`

## 🧠 Model Details

### Algorithm: Random Forest Classifier

**Why Random Forest?**
- **High Accuracy**: Ensemble method reduces overfitting
- **Feature Importance**: Naturally provides feature rankings
- **Robustness**: Handles outliers and missing data well
- **Interpretability**: Easy to understand and explain

**Mathematical Foundation:**
1. **Bootstrap Aggregating**: Each tree trained on random data subset
2. **Feature Randomness**: Random feature selection at each split
3. **Majority Voting**: Final prediction from tree consensus
4. **Confidence Scoring**: Based on vote distribution

### Model Performance

- **Accuracy**: 95%+ on test set
- **Training Data**: 150 samples (Iris dataset)
- **Features**: 4 (sepal length/width, petal length/width)
- **Classes**: 3 (setosa, versicolor, virginica)

### Feature Importance

Based on the trained model:
1. **Petal Length**: Most discriminative feature
2. **Petal Width**: Second most important
3. **Sepal Length**: Moderate importance
4. **Sepal Width**: Least discriminative

## 🛠 Development

### Setting up Development Environment

1. **Install development dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   \`\`\`

2. **Code formatting**
   \`\`\`bash
   black app/ scripts/ test_api.py
   \`\`\`

3. **Linting**
   \`\`\`bash
   flake8 app/ scripts/ test_api.py
   \`\`\`

4. **Type checking**
   \`\`\`bash
   mypy app/
   \`\`\`

### Running in Development Mode

\`\`\`bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
\`\`\`

This enables:
- Auto-reload on code changes
- Detailed error messages
- Debug logging

## 🧪 Testing

### Automated Testing

Run the comprehensive test suite:

\`\`\`bash
python test_api.py
\`\`\`

Tests include:
- ✅ Health check endpoint
- ✅ Single prediction with valid data
- ✅ Batch prediction with multiple samples
- ✅ Input validation and error handling
- ✅ Model information endpoint
- ✅ Edge cases and boundary conditions

### Manual Testing

1. **Interactive API Documentation**
   Visit http://localhost:8000/docs to test endpoints interactively

2. **Command Line Testing**
   \`\`\`bash
   # Test single prediction
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
   
   # Test health check
   curl http://localhost:8000/health
   
   # Test model info
   curl http://localhost:8000/model/info
   \`\`\`

### Test Data Examples

**Iris Setosa (Expected: setosa)**
\`\`\`json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
\`\`\`

**Iris Versicolor (Expected: versicolor)**
\`\`\`json
{"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4}
\`\`\`

**Iris Virginica (Expected: virginica)**
\`\`\`json
{"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
\`\`\`

## 🐳 Deployment

### Docker Deployment

1. **Build the Docker image**
   \`\`\`bash
   docker build -t iris-api .
   \`\`\`

2. **Run the container**
   \`\`\`bash
   docker run -p 8000:8000 iris-api
   \`\`\`

3. **Using Docker Compose**
   \`\`\`bash
   docker-compose up -d
   \`\`\`

### Production Deployment

**Environment Variables:**
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: info)

**Deployment Platforms:**
- **Heroku**: `git push heroku main`
- **AWS ECS**: Use provided Dockerfile
- **Google Cloud Run**: Deploy container directly
- **Azure Container Instances**: One-click deployment

### Scaling Considerations

For production workloads:
- **Load Balancing**: Use multiple API instances
- **Caching**: Cache frequent predictions
- **Model Versioning**: A/B test model updates
- **Monitoring**: Track prediction accuracy and latency
- **Security**: Add authentication and rate limiting

## 📊 Monitoring & Observability

### Health Checks

The API provides comprehensive health monitoring:

\`\`\`bash
curl http://localhost:8000/health
\`\`\`

Response includes:
- Service status
- Model loading status
- Model metadata
- System information

### Logging

The application logs:
- Prediction requests and responses
- Model loading events
- Error conditions
- Performance metrics

### Metrics

Key metrics to monitor:
- **Request Rate**: Predictions per second
- **Response Time**: API latency
- **Error Rate**: Failed predictions
- **Model Accuracy**: Prediction confidence distribution

## 🔧 Configuration

### Model Configuration

Modify `scripts/train_model.py` to adjust:
- **Algorithm**: Change to SVM, Logistic Regression, etc.
- **Hyperparameters**: Tune n_estimators, max_depth, etc.
- **Features**: Add/remove input features
- **Validation**: Change train/test split ratio

### API Configuration

Modify `app/main.py` to adjust:
- **Validation Rules**: Change input ranges
- **Batch Size**: Modify maximum batch size
- **Error Handling**: Customize error responses
- **Documentation**: Update API metadata

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   \`\`\`bash
   git checkout -b feature/amazing-feature
   \`\`\`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the test suite**
   \`\`\`bash
   python test_api.py
   \`\`\`
6. **Commit your changes**
   \`\`\`bash
   git commit -m "Add amazing feature"
   \`\`\`
7. **Push to your branch**
   \`\`\`bash
   git push origin feature/amazing-feature
   \`\`\`
8. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints
- Write comprehensive tests
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI**: For the excellent web framework
- **scikit-learn**: For machine learning capabilities
- **Iris Dataset**: Classic dataset from R.A. Fisher
- **Pydantic**: For data validation and serialization

## 📞 Support

If you encounter any issues or have questions:

1. **Check the documentation**: http://localhost:8000/docs
2. **Run the health check**: http://localhost:8000/health
3. **Review the logs**: Check console output for errors
4. **Open an issue**: Create a GitHub issue with details

## 🚀 What's Next?

Potential enhancements:
- [ ] Add more flower species
- [ ] Implement model versioning
- [ ] Add real-time model retraining
- [ ] Create web interface
- [ ] Add image-based classification
- [ ] Implement A/B testing framework
- [ ] Add comprehensive monitoring dashboard

---

**Happy Classifying! 🌸**

Built with ❤️ using FastAPI and scikit-learn

## Day 2 - Exploring the Project Structure

Today I explored the project folders and checked the Python files to understand how the project is organised.

### 1. Project folders

I found three main folders:

- `app` - Contains the main application file.
- `scripts` - Contains the model training and deployment scripts.
- `venv` - Contains the Python virtual environment.

I also checked the other files, including Dockerfile, requirements.txt, README.md, model metadata and the trained model file.

### 2. Main file

The main file I found is `app/main.py`. It contains the FastAPI application and the endpoints used for predictions and health checks.

### 3. Important Python files

**app/main.py**
- Loads the trained model.
- Takes flower measurements as input.
- Predicts the flower species.
- Provides health check and model information.

**scripts/train_model.py**
- Contains the `train_iris_model()` function.
- Used for training the Iris machine learning model.
- Saves the trained model and its metadata.

**scripts/deployment_demo.py**
- Contains the `demo_workflow()` function.
- Shows the deployment process and Docker commands.
- Includes information about monitoring the application.

### 4. What I understood today

I understood how the project files are organised and how different Python files are used for training the model, making predictions and deployment.

I also learned about functions, classes and modules by exploring the existing code.
