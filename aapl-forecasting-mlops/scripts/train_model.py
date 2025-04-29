import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import mlflow
import mlflow.statsmodels
import os
import warnings
import matplotlib.pyplot as plt # Import matplotlib

warnings.filterwarnings("ignore")

def load_data(file_path):
    """Loads data from a CSV file."""
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    # Ensure the index is a DatetimeIndex and sort it
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Select only the 'Close' column
    return df['AAPL']

def evaluate_model(true_values, predictions):
    """Calculates RMSE and MAE."""
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    mae = mean_absolute_error(true_values, predictions)
    return rmse, mae

def train_and_evaluate(series, p, d, q, return_predictions=False):
    """Trains an ARIMA model, evaluates it, and optionally returns predictions."""
    # Split data (80% train, 20% test)
    train_size = int(len(series) * 0.8)
    train, test = series[0:train_size], series[train_size:len(series)]
    
    history = [x for x in train]
    predictions = list()
    actual_test_values = list() # Store actual test values
    
    try:
        # Walk-forward validation
        for t in range(len(test)):
            model = ARIMA(history, order=(p, d, q))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            actual_test_values.append(obs) # Store actual value
            history.append(obs) # Update history with actual observation
            
        rmse, mae = evaluate_model(test, predictions)
        print(f'ARIMA({p},{d},{q}) - Test RMSE: {rmse:.3f}, MAE: {mae:.3f}')
        
        if return_predictions:
            return rmse, mae, model_fit, test.index, actual_test_values, predictions
        else:
            return rmse, mae, model_fit # Return metrics and the last fitted model
            
    except Exception as e:
        print(f"Error training ARIMA({p},{d},{q}): {e}")
        if return_predictions:
            return float('inf'), float('inf'), None, None, None, None
        else:
            return float('inf'), float('inf'), None


if __name__ == "__main__":
    # Define data path
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'aapl_data.csv')
    
    # Load data
    series = load_data(data_path)
    
    # Define parameter grid for ARIMA (p, d, q) - Keep it small for demonstration
    p_values = [1, 2] # Autoregressive order
    d_values = [1]    # Differencing order (often 1 for non-stationary price data)
    q_values = [1, 2] # Moving average order
    
    best_score, best_cfg, best_mae = float("inf"), None, float("inf")
    best_model_fit_object = None # Store the actual model fit object

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("train_split", 0.8)
        
        print("Starting hyperparameter tuning...")
        # Grid search
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        # Use a nested run for each hyperparameter combination
                        with mlflow.start_run(nested=True):
                            mlflow.log_param("p", p)
                            mlflow.log_param("d", d)
                            mlflow.log_param("q", q)
                            
                            # Pass return_predictions=False during tuning
                            rmse, mae, model_fit = train_and_evaluate(series, p, d, q, return_predictions=False) 
                            
                            mlflow.log_metric("rmse", rmse)
                            mlflow.log_metric("mae", mae)

                            if rmse < best_score and model_fit is not None:
                                best_score, best_cfg, best_mae = rmse, order, mae
                                best_model_fit_object = model_fit # Store the best model object from tuning
                                print(f'>>> New best ARIMA{best_cfg} RMSE={best_score:.3f}, MAE={best_mae:.3f}')

                    except Exception as e:
                        print(f"Failed for order {order}: {e}")
                        continue # Continue to the next combination

        print(f'\nBest ARIMA: {best_cfg} RMSE={best_score:.3f} MAE={best_mae:.3f}')
        
        # Log best parameters and metrics to the parent run
        if best_cfg:
            mlflow.log_param("best_p", best_cfg[0])
            mlflow.log_param("best_d", best_cfg[1])
            mlflow.log_param("best_q", best_cfg[2])
            mlflow.log_metric("best_rmse", best_score)
            mlflow.log_metric("best_mae", best_mae)
            
            # Log the best model artifact using the stored object
            if best_model_fit_object:
                 print("Logging the best model found during tuning...")
                 mlflow.statsmodels.log_model(
                     statsmodels_model=best_model_fit_object,
                     artifact_path="best-arima-model",
                     registered_model_name="ARIMA_AAPL_Forecast" # Optional: Register the model
                 )
                 
                 # --- Generate and Log Plot ---
                 print("Generating and logging prediction plot for the best model...")
                 # Rerun evaluation with best params to get predictions
                 _, _, _, test_index, actual_values, predictions = train_and_evaluate(
                     series, best_cfg[0], best_cfg[1], best_cfg[2], return_predictions=True
                 )

                 if test_index is not None:
                     plt.figure(figsize=(12, 6))
                     plt.plot(test_index, actual_values, label='Actual Test Values')
                     plt.plot(test_index, predictions, label='Predicted Values', linestyle='--')
                     plt.title(f'AAPL Close Price: Actual vs. Predicted (ARIMA{best_cfg})')
                     plt.xlabel('Date')
                     plt.ylabel('Price')
                     plt.legend()
                     plt.grid(True)
                     
                     # Save plot to a file
                     plot_filename = "actual_vs_predicted_test.png"
                     plt.savefig(plot_filename)
                     plt.close() # Close the plot to free memory
                     
                     # Log the plot as an artifact
                     mlflow.log_artifact(plot_filename)
                     print(f"Logged plot: {plot_filename}")
                     # Clean up the saved plot file
                     os.remove(plot_filename) 
                 else:
                     print("Could not generate plot as prediction failed for the best model.")
                 # --- End Plot Generation ---
                 
            else:
                 print("Could not log best model as it was not successfully trained.")
        else:
            print("Hyperparameter tuning did not yield a best model.")

    print("Model training and MLflow logging complete.")

