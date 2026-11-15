from pathlib import Path
import csv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import predict_news
from database import create_table, save_prediction, get_predictions, delete_predictions

app = FastAPI(
    title="Intelligent Fake News Detection and Analysis API",
    description="NLP and machine-learning based news classification API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_table()


class NewsRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)
    category: str = Field(default="", max_length=80)


@app.get("/")
def home():
    return {"message": "Fake News Detection API is running", "status": "success", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: NewsRequest):
    if not request.title.strip() or not request.text.strip():
        raise HTTPException(status_code=422, detail="Title and article content cannot be empty.")
    try:
        result = predict_news(request.title, request.text, request.category)
        save_prediction(
            title=request.title.strip(),
            text=request.text.strip(),
            category=request.category,
            prediction=result["prediction"],
            confidence=result["confidence"],
            fake_probability=result["fake_probability"],
            real_probability=result["real_probability"],
        )
        return {"success": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction could not be completed: {exc}")


@app.get("/history")
def history():
    records = get_predictions()
    return {"success": True, "count": len(records), "history": records}


@app.delete("/history")
def clear_history():
    delete_predictions()
    return {"success": True, "message": "Prediction history cleared."}


@app.get("/metrics")
def model_metrics():
    report_path = Path(__file__).resolve().parent / "model_comparison_results.csv"
    models = []
    if report_path.exists():
        with report_path.open("r", newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                try:
                    models.append({
                        "model": row["Model"],
                        "accuracy": float(row["Accuracy"]),
                        "precision": float(row["Precision"]),
                        "recall": float(row["Recall"]),
                        "f1_score": float(row["F1 Score"]),
                    })
                except (KeyError, ValueError):
                    continue
    return {"success": True, "models": models}
