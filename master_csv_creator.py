import pandas as pd
import os
from datetime import datetime
import glob

def read_and_combine_csvs():
    # Get all CSV files from the polymarket_data_csvs directory
    csv_files = glob.glob('polymarket_data_csvs/*.csv')
    
    # Initialize an empty list to store all dataframes
    all_dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        df = pd.read_csv(file)
        # Convert date column to datetime
        df['Date (UTC)'] = pd.to_datetime(df['Date (UTC)'])
        all_dfs.append(df)
    
    # Combine all dataframes
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by date
    master_df = master_df.sort_values('Date (UTC)')
    
    # Save the time-series master CSV
    master_df.to_csv('master_time_series.csv', index=False)
    
    # Create weekly standardized view
    # Group by week and calculate mean for each price range
    weekly_df = master_df.copy()
    weekly_df['Week'] = weekly_df['Date (UTC)'].dt.isocalendar().week
    weekly_df['Year'] = weekly_df['Date (UTC)'].dt.isocalendar().year
    
    # Group by year and week, calculate mean for each price range
    price_columns = [col for col in weekly_df.columns if col not in ['Date (UTC)', 'Timestamp (UTC)', 'Week', 'Year']]
    weekly_aggregated = weekly_df.groupby(['Year', 'Week'])[price_columns].mean().reset_index()
    
    # Save the weekly standardized view
    weekly_aggregated.to_csv('master_weekly_standardized.csv', index=False)
    
    print("Successfully created:")
    print("1. master_time_series.csv - Complete time series data")
    print("2. master_weekly_standardized.csv - Weekly standardized view")

if __name__ == "__main__":
    read_and_combine_csvs()
