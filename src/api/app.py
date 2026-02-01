"""FastAPI app for predictions."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_matchup

app = FastAPI(title="UFC Predictor")


class PredictRequest(BaseModel):
    fighter_a: str
    fighter_b: str
    date: str
    model_path: str = "artifacts/model_logistic.joblib"


@app.post("/predict")
async def predict(req: PredictRequest) -> dict:
    result = predict_matchup(
        req.fighter_a,
        req.fighter_b,
        req.date,
        Path(req.model_path),
    )
    return result


