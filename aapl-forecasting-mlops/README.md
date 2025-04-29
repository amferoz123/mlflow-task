# AAPL Forecasting MLOps Project

This project is a class assignment designed to provide hands-on experience with machine learning operations (MLOps) techniques using MLflow and deployment on AWS. The goal is to develop a time series forecasting solution for predicting Apple (AAPL) stock prices using the ARIMA model.

## Project Structure

```
aapl-forecasting-mlops
├── data
│   ├── raw                # Raw data collected from yfinance
│   └── processed          # Processed data ready for model training
├── notebooks
│   └── exploration.ipynb  # Jupyter notebook for exploratory data analysis
├── scripts
│   ├── fetch_data.py      # Script to fetch stock price data
│   └── train_model.py     # Script to train the ARIMA model with hyperparameter tuning
├── src
│   ├── __init__.py        # Marks src as a Python package
│   ├── api
│   │   ├── __init__.py    # Marks api as a Python package
│   │   ├── main.py        # Entry point for the FastAPI application
│   │   └── endpoints.py    # Defines FastAPI endpoints
│   ├── data_processing.py  # Functions for data preprocessing
│   ├── model.py           # Implements the ARIMA model
│   └── utils.py           # Utility functions
├── tests
│   ├── __init__.py        # Marks tests as a Python package
│   ├── test_api.py        # Unit tests for FastAPI endpoints
│   └── test_model.py      # Unit tests for model functionalities
├── .gitignore              # Specifies files to be ignored by Git
├── Dockerfile              # Instructions for building a Docker image
├── requirements.txt        # Lists Python dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/aapl-forecasting-mlops.git
   cd aapl-forecasting-mlops
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1.  **Data Collection**: Run the `fetch_data.py` script to collect historical stock price data for Apple (AAPL):
    ```bash
    python scripts/fetch_data.py
    ```

2.  **Model Training**: Train the ARIMA model by executing the `train_model.py` script. This script performs hyperparameter tuning and logs experiments to MLflow:
    ```bash
    python scripts/train_model.py
    ```
    *Ensure MLflow tracking server is running or configured if not using local tracking (`./mlruns` directory should be created).*

3.  **Exploratory Data Analysis**: Use the Jupyter notebook for EDA:
    ```bash
    jupyter notebook notebooks/exploration.ipynb
    ```

4.  **API Development**: Start the FastAPI application using Uvicorn. Run this command from the project root directory (`aapl-forecasting-mlops`):
    ```bash
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
    ```

5.  **Access the API**:
    *   The API documentation (Swagger UI) will be available at `http://127.0.0.1:8000/docs`.
    *   The health check endpoint is at `http://127.0.0.1:8000/api/health`.
    *   The prediction endpoint is at `http://127.0.0.1:8000/api/predict`. Example usage: `http://127.0.0.1:8000/api/predict?steps=5` to forecast the next 5 steps.

## MLflow Tracking

This project utilizes MLflow for experiment tracking. The `train_model.py` script performs the following:
*   Loads the fetched AAPL closing price data.
*   Splits the data into training (80%) and testing (20%) sets.
*   Performs hyperparameter tuning for the ARIMA(p,d,q) model using a grid search approach.
*   Logs hyperparameters (p, d, q), performance metrics (RMSE, MAE), and model artifacts for each run.
*   Identifies the best performing model based on RMSE on the test set.
*   Logs the best hyperparameters and metrics.
*   Logs the best ARIMA model artifact using `mlflow.statsmodels.log_model`.
*   Optionally registers the best model in the MLflow model registry under the name "ARIMA\_AAPL\_Forecast".

## Deployment

The application can be deployed on an AWS EC2 instance (e.g., `t3.micro`).

### AWS EC2 Deployment Steps

1.  **Launch EC2 Instance**:
    *   Navigate to the AWS EC2 console and launch a new instance.
    *   Choose an AMI (e.g., Ubuntu Server).
    *   Select instance type `t3.micro`.
    *   Configure instance details (defaults are often sufficient).
    *   Add storage (default is usually okay).
    *   **Configure Security Group**: Create or select a security group. Ensure the following inbound rules are added:
        *   `SSH` (TCP port 22) from `My IP` or your specific IP range for secure access.
        *   `Custom TCP` (TCP port 8000, or your API port) from `Anywhere` (`0.0.0.0/0`, `::/0`) for public API access. *Restrict source IPs in production if possible.*
    *   Review and launch, selecting or creating a key pair for SSH access.

2.  **Connect to EC2**:
    *   Use SSH with your key pair and the instance's public IP/DNS:
        ```bash
        ssh -i /path/to/your-key.pem ubuntu@<your-ec2-public-ip>
        ```

3.  **Set up Environment**:
    *   Update packages: `sudo apt update`
    *   Install Python, pip, venv: `sudo apt install python3 python3-pip python3-venv -y`
    *   Install Git: `sudo apt install git -y`

4.  **Deploy Code**:
    *   Clone the repository: `git clone https://github.com/yourusername/aapl-forecasting-mlops.git`
    *   Change directory: `cd aapl-forecasting-mlops`

5.  **Install Dependencies**:
    *   Create and activate virtual environment:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   Install packages: `pip install -r requirements.txt`

6.  **Configure MLflow Model Access**:
    *   **Local `./mlruns`**: If you trained the model locally and the `./mlruns` directory contains the model artifact, ensure this directory is copied to the EC2 instance within the project structure (e.g., via Git if small, or using `scp`). The API will look for it locally by default.
    *   **Remote MLflow Server**: If you used a remote MLflow tracking server, set the environment variable *before* starting the API:
        ```bash
        export MLFLOW_TRACKING_URI='http://your-mlflow-server-ip:5000'
        ```

7.  **Run the API**:
    *   Start Uvicorn from the project root directory:
        ```bash
        uvicorn src.api.main:app --host 0.0.0.0 --port 8000
        ```
    *   For persistent background execution, use `screen`, `tmux`, or configure a `systemd` service.

8.  **Access Public API**:
    *   Use the instance's public IP or DNS in your browser/client.
    *   Health: `http://<your-ec2-public-ip>:8000/api/health`
    *   Predict: `http://<your-ec2-public-ip>:8000/api/predict?steps=N`
    *   Docs: `http://<your-ec2-public-ip>:8000/docs`
