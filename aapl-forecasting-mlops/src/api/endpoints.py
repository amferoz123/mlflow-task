from fastapi import APIRouter, HTTPException, Query
import mlflow
import pandas as pd
import logging
import os
from pathlib import Path # Import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Model Loading ---
MODEL_NAME = "ARIMA_AAPL_Forecast"
MODEL_STAGE = "None" # Or "Staging", "Production" if using stages

# --- Set Tracking URI ---
# Calculate the path to the mlruns directory relative to the project root
# Assumes the API is run from the project root directory
project_root = Path(__file__).parent.parent.parent # Get project root (aapl-forecasting-mlops)
local_mlruns_path = project_root / "scripts" / "mlruns"
local_tracking_uri = f"file://{local_mlruns_path.resolve()}"

loaded_model = None

try:
    # Explicitly set the tracking URI if MLFLOW_TRACKING_URI is not set
    if "MLFLOW_TRACKING_URI" not in os.environ:
        logger.info(f"MLFLOW_TRACKING_URI not set. Setting local tracking URI to: {local_tracking_uri}")
        mlflow.set_tracking_uri(local_tracking_uri)
    else:
        logger.info(f"Using MLFLOW_TRACKING_URI from environment: {os.environ['MLFLOW_TRACKING_URI']}")

    # Construct the model URI (remains the same, MLflow uses the set tracking URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info(f"Attempting to load model '{MODEL_NAME}' stage '{MODEL_STAGE}' using tracking URI: {mlflow.get_tracking_uri()}")

    # Load the model using mlflow.statsmodels.load_model
    loaded_model = mlflow.statsmodels.load_model(model_uri)
    logger.info(f"Successfully loaded model: {loaded_model}")
    # logger.info(f"Model summary: {loaded_model.summary()}") # Can be verbose

except Exception as e:
    logger.error(f"Failed to load model '{MODEL_NAME}' from stage '{MODEL_STAGE}'. Tracking URI: {mlflow.get_tracking_uri()}. Error: {e}", exc_info=True)
    loaded_model = None

# --- Endpoints ---

@router.get("/health", summary="Health Check", status_code=200)
async def health_check():
    """
    Simple health check endpoint to confirm the API is running.
    """
    # Basic check: API is up. More advanced checks could verify model loading, DB connections etc.
    status = "OK" if loaded_model is not None else "ERROR: Model not loaded"
    return {"status": status}

@router.get("/predict", summary="Predict Future Stock Prices")
async def predict(
    steps: int = Query(..., gt=0, description="Number of future steps to forecast")
):
    """
    Predicts future AAPL closing prices using the loaded ARIMA model.

    - **steps**: Number of future time steps to predict (must be greater than 0).
    """
    if loaded_model is None:
        logger.error("Prediction attempt failed: Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not available. Service Unavailable.")

    try:
        logger.info(f"Received prediction request for {steps} steps.")
        # Use the forecast method of the loaded statsmodels ARIMAResultsWrapper
        forecast_values = loaded_model.forecast(steps=steps)
        logger.info(f"Generated forecast: {forecast_values}")

        # Format the output (e.g., as a list or dictionary)
        # Creating a simple list of predicted values
        predictions = forecast_values.tolist()

        return {"forecast_steps": steps, "predictions": predictions}

    except Exception as e:
        logger.error(f"Prediction failed for {steps} steps. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

