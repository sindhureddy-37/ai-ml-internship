
from fastapi import FastAPI

app = FastAPI(title="AI/ML Internship Project")

@app.get("/")
def home():
    return {
        "message": "AI/ML Internship application is running!",
        "status": "success",
        "project": "Week 1 - Day 1"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
