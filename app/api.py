from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.predictor import SentimentPredictor

router = APIRouter()
predictor = SentimentPredictor()

class SummaryInput(BaseModel):
    summary: str

@router.post("/predict")
def predict_sentiment(data: SummaryInput):
    try:
        sentiment = predictor.predict(data.summary)
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
