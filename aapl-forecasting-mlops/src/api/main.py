from fastapi import FastAPI
from .endpoints import router as api_router
import logging

# Configure logging (optional, can inherit from endpoints or set globally)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AAPL Stock Price Forecasting API",
    description="API for predicting Apple (AAPL) stock prices using an ARIMA model trained with MLflow.",
    version="0.1.0",
)

# Include the router from endpoints.py
app.include_router(api_router, prefix="/api", tags=["Forecasting"]) # Add prefix for versioning/scoping

@app.get("/", summary="Root Endpoint", tags=["General"])
async def read_root():
    """
    Root endpoint providing basic information about the API.
    """
    return {"message": "Welcome to the AAPL Forecasting API. Visit /docs for details."}

