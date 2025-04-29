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
│   └── train_model.py     # Script to train the ARIMA model
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

1. **Data Collection**: Run the `fetch_data.py` script to collect historical stock price data for Apple (AAPL):
   ```
   python scripts/fetch_data.py
   ```

2. **Model Training**: Train the ARIMA model by executing the `train_model.py` script:
   ```
   python scripts/train_model.py
   ```

3. **Exploratory Data Analysis**: Use the Jupyter notebook for EDA:
   ```
   jupyter notebook notebooks/exploration.ipynb
   ```

4. **API Development**: Start the FastAPI application:
   ```
   uvicorn src.api.main:app --reload
   ```

5. **Access the API**: The API will be available at `http://127.0.0.1:8000`. You can access the `/predict` and `/health` endpoints.

## MLflow Tracking

This project utilizes MLflow for experiment tracking. All model hyperparameters, performance metrics (RMSE, MAE), and model artifacts are logged during the training process. The best-performing model is registered in the MLflow model registry.

## Deployment

The application can be deployed on an AWS EC2 t3.micro instance. Ensure to configure security groups for public API accessibility.
