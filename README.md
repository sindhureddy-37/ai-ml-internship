
# AI/ML Internship - Week 1 Day 1

## Project: Clone & Run 60

### Objective
Set up a local AI/ML development environment and run FastAPI and MLflow.

### Environment
- Python 3.14.6
- Virtual Environment: venv
- MLflow 3.18.0
- FastAPI
- Uvicorn
- Scikit-learn
- Docker 29.9.2

### Completed Tasks
- Created project workspace.
- Created and activated Python virtual environment.
- Installed required Python packages.
- Verified Docker installation.
- Created a FastAPI application.
- Started FastAPI on port 8000.
- Started MLflow UI on port 5000.
- Verified both applications in the browser.

### Run FastAPI
```powershell
uvicorn app:app --reload
```

### Run MLflow
```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

### Application URLs
- FastAPI: http://127.0.0.1:8000
- Health Check: http://127.0.0.1:8000/health
- MLflow: http://127.0.0.1:5000

### Status
Local application setup test completed successfully.
