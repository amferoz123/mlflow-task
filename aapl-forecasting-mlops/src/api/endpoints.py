from fastapi import APIRouter, HTTPException, Query
import mlflow
from mlflow.tracking import MlflowClient # Import MlflowClient
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository # Import ModelsArtifactRepository
import statsmodels.api as sm # Import statsmodels for direct loading
import pandas as pd
import logging
import os
from pathlib import Path

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

    # Construct the model URI
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info(f"Attempting to load model '{MODEL_NAME}' stage '{MODEL_STAGE}' using tracking URI: {mlflow.get_tracking_uri()}")

    # --- Resolve artifact path dynamically ---
    # Use ModelsArtifactRepository to get the local path where the artifact should be.
    # It handles downloading/resolving based on the currently set tracking URI.
    # The empty string "" signifies downloading the root artifact directory for the model version.
    logger.info(f"Resolving artifact path for model URI: {model_uri}")
    model_artifact_path = ModelsArtifactRepository(model_uri).download_artifacts("")
    logger.info(f"Artifacts resolved/downloaded to local path: {model_artifact_path}")

    # Construct the path to the actual model file (usually model.pkl for statsmodels)
    # Check common filenames used by mlflow.statsmodels.log_model
    potential_model_filenames = ["model.pkl", "model.pickle"]
    model_file_path = None
    for filename in potential_model_filenames:
        path_check = os.path.join(model_artifact_path, filename)
        if os.path.exists(path_check):
            model_file_path = path_check
            logger.info(f"Found model file at: {model_file_path}")
            break
    
    if not model_file_path:
         # Check if the artifact path itself is the model file (less common for statsmodels)
         if os.path.isfile(model_artifact_path) and model_artifact_path.endswith(('.pkl', '.pickle')):
             model_file_path = model_artifact_path
             logger.info(f"Found model file directly at artifact path: {model_file_path}")
         else:
             # If still not found, check the 'best-arima-model' subdirectory if it exists
             # This matches the artifact_path used in train_model.py
             subdir_path = os.path.join(model_artifact_path, "best-arima-model")
             if os.path.isdir(subdir_path):
                 logger.info(f"Checking subdirectory: {subdir_path}")
                 for filename in potential_model_filenames:
                     path_check = os.path.join(subdir_path, filename)
                     if os.path.exists(path_check):
                         model_file_path = path_check
                         logger.info(f"Found model file in subdirectory: {model_file_path}")
                         break
             
             if not model_file_path:
                 logger.error(f"Could not find model file (e.g., model.pkl) within resolved artifact path: {model_artifact_path} or its 'best-arima-model' subdirectory.")
                 raise FileNotFoundError(f"Model file not found in {model_artifact_path}")

    # Load the model directly using statsmodels' load function
    logger.info(f"Loading model directly from: {model_file_path}")
    loaded_model = sm.load(model_file_path)
    logger.info(f"Successfully loaded model directly from pickle file.")
    # logger.info(f"Model summary: {loaded_model.summary()}") # Can be verbose

except Exception as e:
    logger.error(f"Failed to load model '{MODEL_NAME}' from stage '{MODEL_STAGE}'. Tracking URI: {mlflow.get_tracking_uri()}. Error: {e}", exc_info=True)
    loaded_model = None # Ensure model is None on failure

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

