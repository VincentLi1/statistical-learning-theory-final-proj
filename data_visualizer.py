import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def create_visualizations():
    # Create output directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Read the master files
    time_series_df = pd.read_csv('master_time_series.csv')
    weekly_df = pd.read_csv('master_weekly_standardized.csv')
    
    # Convert date columns to datetime
    time_series_df['Date (UTC)'] = pd.to_datetime(time_series_df['Date (UTC)'])
    
    # Set style to default
    plt.style.use('default')
    
    # 1. Time Series Plot of All Price Ranges
    plt.figure(figsize=(15, 8))
    price_columns = [col for col in time_series_df.columns if col not in ['Date (UTC)', 'Timestamp (UTC)']]
    for column in price_columns:
        plt.plot(time_series_df['Date (UTC)'], time_series_df[column], label=column)
    
    plt.title('Polymarket Price Ranges Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/time_series_all_ranges.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Weekly Heatmap
    plt.figure(figsize=(15, 8))
    weekly_heatmap = weekly_df.drop(['Year', 'Week'], axis=1)
    sns.heatmap(weekly_heatmap.T, cmap='YlOrRd', annot=True, fmt='.2f')
    plt.title('Weekly Price Range Heatmap')
    plt.xlabel('Week')
    plt.ylabel('Price Range')
    plt.tight_layout()
    plt.savefig('visualizations/weekly_heatmap.png', dpi=300)
    plt.close()
    
    # 3. Box Plot of Price Ranges
    plt.figure(figsize=(15, 8))
    time_series_df.boxplot(column=price_columns)
    plt.title('Distribution of Price Ranges')
    plt.xlabel('Price Range')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/price_range_boxplot.png', dpi=300)
    plt.close()
    
    # 4. Weekly Trend Lines
    plt.figure(figsize=(15, 8))
    for column in price_columns:
        plt.plot(weekly_df.index, weekly_df[column], label=column, marker='o')
    
    plt.title('Weekly Price Range Trends')
    plt.xlabel('Week')
    plt.ylabel('Average Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/weekly_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations created in the 'visualizations' directory:")
    print("1. time_series_all_ranges.png - Complete time series of all price ranges")
    print("2. weekly_heatmap.png - Heatmap showing weekly price distributions")
    print("3. price_range_boxplot.png - Box plot showing price range distributions")
    print("4. weekly_trends.png - Weekly trend lines for each price range")

if __name__ == "__main__":
    create_visualizations() 