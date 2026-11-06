# AI/ML Internship - Week 1 Day 1

## Project: Clone & Run 60

### Objective
Set up an AI/ML environment and run an open-source machine learning API.

### Environment
- Python 3.14.6 (initial setup)
- Python 3.11.9 (Iris project)
- MLflow 3.18.0
- FastAPI
- Uvicorn
- Scikit-learn
- Docker 29.9.2

### Completed Tasks
- Created project workspace.
- Created Python virtual environments.
- Installed required packages.
- Verified Docker CLI.
- Started FastAPI application.
- Started MLflow UI.
- Cloned an open-source Iris project.
- Loaded the trained machine learning model.
- Tested the health endpoint successfully.
- Tested the prediction endpoint successfully.
- Predicted Iris Setosa with 100% confidence.

### Open-Source Repository
https://github.com/Prabhu6626/Deploying-Machine-Learning-Models-with-FastAPI

### Run Iris API
```powershell
cd Deploying-Machine-Learning-Models-with-FastAPI
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8001