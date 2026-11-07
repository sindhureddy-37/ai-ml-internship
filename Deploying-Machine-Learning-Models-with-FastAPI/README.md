
# Iris Classification API with FastAPI

A machine learning API for classifying Iris flower species using FastAPI and scikit-learn. This project demonstrates how to deploy a machine learning model as a REST API.

## Table of Contents

1. Overview
2. Features
3. Installation and Quick Start
4. API Documentation
5. Project Structure
6. Model Details
7. Testing
8. Docker Deployment
9. Monitoring
10. Day 1 - Project Setup and Environment Configuration
11. Day 2 - Exploring the Project Structure
12. Day 3 - Understanding the Data Pipeline

## 1. Overview

This project classifies Iris flowers into three species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

The API accepts four flower measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

It returns the predicted flower species along with confidence and probability values.

## 2. Features

- FastAPI-based REST API.
- Random Forest machine learning model.
- Input validation using Pydantic.
- Single flower prediction.
- Batch prediction for up to 100 samples.
- Health check endpoint.
- Model information endpoint.
- Docker deployment support.
- Logging and error handling.

## 3. Installation and Quick Start

### Prerequisites

- Python 3.9 or above
- pip
- Git
- Docker (for containerisation)

### Installation

**1. Create a virtual environment**

```bash
python -m venv venv
```

**2. Activate the virtual environment**

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Train the model**

```bash
python scripts/train_model.py
```

**5. Start the API server**

```bash
uvicorn app.main:app --reload
```

**6. Open the API documentation**

```text
http://localhost:8000/docs
```

The API is available at:

```text
http://localhost:8000
```

## 4. API Documentation

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction |
| GET | `/docs` | Interactive API documentation |

### Input Example

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Output Example

```json
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
```

### Input Validation

The API uses the `IrisFeatures` Pydantic class to validate input.

- Measurements must be numeric.
- Values must be between 0.1 and 10.0.
- Invalid input is rejected by the API.

## 5. Project Structure

```text
Deploying-Machine-Learning-Models-with-FastAPI/
│
├── app/
│   └── main.py
│
├── scripts/
│   ├── train_model.py
│   ├── setup.sh
│   └── deployment_demo.py
│
├── venv/
├── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── iris_model.joblib
└── model_metadata.json
```

## 6. Model Details

### Algorithm: Random Forest Classifier

The project uses a Random Forest classifier to predict the Iris flower species.

The model works with four input features and predicts one of three flower classes.

### Model Information

- Algorithm: Random Forest
- Dataset: Iris dataset
- Number of samples: 150
- Input features: 4
- Output classes: 3

The model achieved approximately 96.67% accuracy in the training and testing work documented during Day 1.

## 7. Testing

Run the test script using:

```bash
python test_api.py
```

The API can also be tested through the interactive documentation:

```text
http://localhost:8000/docs
```

## 8. Docker Deployment

Docker is used to package the application and its dependencies.

### Build the Docker image

```bash
docker build -t iris-api .
```

### Run the Docker container

```bash
docker run -p 8000:8000 iris-api
```

### Docker Compose

```bash
docker-compose up -d
```

## 9. Monitoring

The application includes monitoring-related features.

### Health Checks

The `/health` endpoint provides information about the API and model status.

### Logging

The application uses logging to record important events, prediction activities and errors.

### Important Metrics

- API response time
- Request rate
- Error rate
- Model prediction behaviour

---

## Day 1 - Project Setup and Environment Configuration

Today I started working on the machine learning deployment project and prepared my development environment.

### 1. Python Setup

- Installed Python on my Windows laptop.
- Checked the Python installation and version.
- Learned how Python is used in machine learning projects.

### 2. Virtual Environment

- Created a Python virtual environment.
- Activated the virtual environment.
- Understood why virtual environments are used to manage project dependencies separately.

### 3. Installing Required Tools and Libraries

Installed and explored the tools required for the project:

- FastAPI - To build the machine learning API.
- Uvicorn - To run the FastAPI application.
- scikit-learn - To work with machine learning models.
- MLflow - To explore machine learning experiment tracking.
- Docker - To understand application containerisation.

### 4. Running the Application

- Explored how to start a FastAPI application using Uvicorn.
- Ran the application locally.
- Opened the application in a browser.
- Checked the API documentation and endpoints.

### 5. MLflow and Docker

- Explored how MLflow is used for experiment tracking.
- Learned how Docker packages applications and dependencies.
- Checked the availability of the required tools.

### 6. What I Learned Today

I learned how to set up a Python development environment, create and activate a virtual environment, install the required libraries and run a FastAPI application.

I also gained an introduction to MLflow, Docker and machine learning model deployment.

---

## Day 2 - Exploring the Project Structure

Today I explored the project folders and checked the Python files to understand how the project is organised.

### 1. Project Folders

I identified the following folders:

- `app` - Contains the main application file.
- `scripts` - Contains the model training and deployment scripts.
- `venv` - Contains the Python virtual environment.

I also explored other project files, including the Dockerfile, requirements.txt, README.md, model metadata and trained model file.

### 2. Main File

The main file is `app/main.py`.

It contains the FastAPI application and the endpoints used for predictions and health checks.

### 3. Important Python Files

**app/main.py**

- Loads the trained model.
- Accepts flower measurements.
- Predicts the flower species.
- Provides health check and model information.

**scripts/train_model.py**

- Contains the `train_iris_model()` function.
- Trains the Iris machine learning model.
- Saves the trained model and its metadata.

**scripts/deployment_demo.py**

- Contains the `demo_workflow()` function.
- Demonstrates the deployment workflow.
- Includes information about Docker and monitoring.

### 4. What I Learned Today

I understood how the project files are organised and how different Python files are used for training the model, making predictions and deployment.

I also learned about functions, classes and modules by exploring the existing code.

---

## Day 3 - Understanding the Data Pipeline

Today I explored how data enters the Iris Classification API, how it is processed by the machine learning model, and what output the application produces.

### 1. Input Data

The project accepts four numerical measurements of an Iris flower.

| Feature | Data Type |
|---|---|
| Sepal length | Float |
| Sepal width | Float |
| Petal length | Float |
| Petal width | Float |

The `IrisFeatures` Pydantic class validates the input data.

- All measurements must be numeric.
- Values must be between 0.1 and 10.0.
- Invalid inputs are rejected.

The input is received in JSON format through the prediction API.

### 2. Data Processing

The data passes through the following steps:

1. FastAPI receives the flower measurements.
2. Pydantic validates the input using the `IrisFeatures` class.
3. The measurements are prepared as a model input array.
4. The trained Random Forest model processes the input.
5. `model.predict()` predicts the flower species.
6. `model.predict_proba()` calculates the probabilities for each species.

The application also supports batch prediction using a Python list of input samples. It allows up to 100 samples in one batch request.

### 3. Output Data

The API returns a prediction response containing:

- `predicted_class` - The predicted Iris flower species.
- `confidence` - The confidence score of the prediction.
- `probabilities` - The probability of each flower species.
- `input_features` - The original flower measurements.

The response is returned in JSON format.

### 4. Data Flow Diagram

The following diagram shows how data moves through the project:

```text
          USER / CLIENT
                |
                v
       FLOWER MEASUREMENTS
           (JSON Input)
                |
                v
         FASTAPI ENDPOINT
                |
                v
       PYDANTIC VALIDATION
                |
                v
         MODEL INPUT ARRAY
                |
                v
       TRAINED RANDOM FOREST
              MODEL
                |
          +-----+-----+
          |           |
          v           v
      PREDICTION   PROBABILITIES
          |           |
          +-----+-----+
                |
                v
        API RESPONSE (JSON)
                |
                v
          USER / CLIENT
```

### 5. End-to-End MLOps

MLOps combines machine learning development and software operations to manage the machine learning lifecycle.

**1. Experiment Tracking**

- Records different model experiments.
- Tracks model parameters and performance.
- Helps compare different model versions.
- MLflow is an example of an experiment tracking tool.

**2. Containerisation**

- Packages the application and its dependencies together.
- Docker is used to create containers.
- Makes application deployment more consistent across environments.

Example commands:

```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```

**3. CI/CD (Continuous Integration and Continuous Deployment)**

- CI checks code changes and runs automated tests.
- CD automates the deployment process.
- Helps maintain code quality and reliable releases.
- GitHub Actions can be used to automate CI/CD workflows.

**4. Model Serving**

- Makes trained machine learning models available through APIs.
- FastAPI serves the Iris classification model.
- Users send flower measurements and receive predictions.
- The application supports single and batch predictions.

**5. Monitoring**

- Health checks help monitor API and model availability.
- Logging records application events and errors.
- Response time and error rates help monitor API performance.
- Model performance can be evaluated over time.

### 6. What I Learned Today

Today I understood how data enters the project, how Pydantic validates the input, how the model processes the measurements and how the API returns predictions.

I also learned about arrays, lists and dictionaries used in machine learning applications.

I gained an understanding of end-to-end MLOps, including experiment tracking, Docker containerisation, CI/CD, model serving and monitoring.

---

## Conclusion

Through this project, I learned about setting up a Python development environment, deploying a machine learning model using FastAPI, exploring project structure, understanding data pipelines and learning the basic concepts of MLOps.

Built with using FastAPI and scikit-learn.
