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
    start_date = '2020-01-01'
    end_date = '2025-01-01'
    
    # Fetch the data
    aapl_data = fetch_aapl_data(start_date, end_date)
    
   #create data directory if it doesn't exist one level up from the script
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Define the file path to save the data
    file_path = os.path.join(data_dir, 'aapl_data.csv')
    
    # Save the data to a CSV file
    save_data_to_csv(aapl_data, file_path)
    
    print(f"Data saved to {file_path}")
    print("Data fetching complete.")