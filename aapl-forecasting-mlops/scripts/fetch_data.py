import yfinance as yf
import pandas as pd
import os

def fetch_aapl_data(start_date, end_date):
    # Fetch historical closing prices for Apple (AAPL)
    aapl_data = yf.download('AAPL', start=start_date, end=end_date)
    return aapl_data['Close']

def save_data_to_csv(data, file_path):
    # Save the fetched data to a CSV file
    data.to_csv(file_path, index=True)

if __name__ == "__main__":
    # Define the date range for data collection
    start_date = '2023-01-01'
    end_date = '2025-01-01'
    
    # Fetch the data
    aapl_data = fetch_aapl_data(start_date, end_date)
    
    # Create the raw data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Save the data to the raw data directory
    save_data_to_csv(aapl_data, 'data/raw/aapl_stock_data.csv')