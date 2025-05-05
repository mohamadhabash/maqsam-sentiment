from fastapi import FastAPI
from app.api import router as sentiment_router

app = FastAPI(title="Maqsam Sentiment Classifier")
app.include_router(sentiment_router, prefix="/api", tags=["Sentiment"])
