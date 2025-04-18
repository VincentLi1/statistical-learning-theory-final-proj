import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import traceback
from sklearn.preprocessing import StandardScaler
import glob

def load_data():
    """Load and prepare the data from CSV files"""
    print("\nLoading and preparing data...")
    
    # Load minute-by-minute data from recent day
    print("\nLoading minute-by-minute data...")
    time_series = pd.read_csv('polymarket_data_csvs/polymarket-price-recent-day-minute-by-minute.csv')
    time_series['Date (UTC)'] = pd.to_datetime(time_series['Date (UTC)'])
    time_series = time_series.sort_values('Date (UTC)')
    print(f"Time series data shape: {time_series.shape}")
    
    # Load sentiment data
    print("\nLoading sentiment data...")
    sentiment_data = pd.read_csv('musksentiment_weekly.csv')
    print(f"Sentiment data shape: {sentiment_data.shape}")
    
    # Convert date columns to datetime
    print("\nConverting dates...")
    sentiment_data['week'] = pd.to_datetime(sentiment_data['week'])
    
    # Extract year and week from dates
    time_series['Year'] = time_series['Date (UTC)'].dt.isocalendar().year
    time_series['Week'] = time_series['Date (UTC)'].dt.isocalendar().week
    
    sentiment_data['Year'] = sentiment_data['week'].dt.isocalendar().year
    sentiment_data['Week'] = sentiment_data['week'].dt.isocalendar().week
    
    # Filter sentiment data to match market data year
    sentiment_data = sentiment_data[sentiment_data['Year'] == 2025]
    print(f"\nFiltered sentiment data shape: {sentiment_data.shape}")
    
    # Get price columns (excluding date and metadata columns)
    price_columns = [col for col in time_series.columns if col not in ['Date (UTC)', 'Timestamp (UTC)', 'Year', 'Week']]
    
    print(f"\nNumber of price columns: {len(price_columns)}")
    print("Sample price columns:", price_columns[:5])
    
    return time_series, sentiment_data, price_columns

def analyze_price_columns(time_series):
    """Analyze price columns to identify those with sufficient data"""
    print("\nAnalyzing price columns...")
    
    # Get price columns (excluding date and metadata columns)
    price_columns = [col for col in time_series.columns if col not in ['Date (UTC)', 'Year', 'Week']]
    
    # Sort price columns to ensure consistent order
    price_columns = sorted(price_columns)
    
    valid_columns = []
    
    print("\nData availability analysis:")
    for col in price_columns:
        # Calculate data availability
        data_ratio = time_series[col].notna().mean()
        print(f"{col}: {data_ratio*100:.1f}% data available")
        
        # Select columns with sufficient data
        if data_ratio >= 0.1:  # More lenient threshold to include more data
            valid_columns.append(col)
    
    if not valid_columns:
        raise ValueError("No valid price columns found with sufficient data")
        
    print(f"\nSelected {len(valid_columns)} price columns with sufficient data")
    print("\nSample of valid columns:", valid_columns[:5])
    
    return valid_columns

def calculate_trend(df, col):
    """Calculate week-over-week trend for a column"""
    try:
        # Calculate percentage change with explicit fill_method=None
        trend = df[col].pct_change(fill_method=None)
        # Replace inf and -inf with 0
        trend = trend.replace([np.inf, -np.inf], 0)
        # Fill NaN with 0
        trend = trend.fillna(0)
        # Clip extreme values
        trend = trend.clip(-1, 1)  # Limit trends to -100% to +100%
        return trend
    except Exception as e:
        print(f"Error calculating trend for {col}: {str(e)}")
        return pd.Series(0, index=df.index)

def create_features_and_target(market_data, sentiment_data, valid_columns):
    """Create features and target variables from market and sentiment data"""
    try:
        # Create features DataFrame
        features = pd.DataFrame()
        features['Week'] = market_data['Week']
        
        # Merge with sentiment data
        features = features.merge(sentiment_data[['Week', 'vader_mean']], on='Week', how='left')
        features['vader_mean'] = features['vader_mean'].fillna(0)
        
        print("\nCalculating trends...")
        # Calculate trends for each valid price column
        for col in valid_columns:
            if col != 'Week' and col != 'Timestamp (UTC)':
                trend_col = f'trend_{col}'
                features[trend_col] = calculate_trend(market_data, col)
        
        # Create target variables (next week's prices)
        targets = market_data[valid_columns].copy()
        
        # Forward fill missing values first
        targets = targets.ffill()
        
        # Then backward fill any remaining NaN values
        targets = targets.bfill()
        
        # Shift targets to get next week's values
        targets = targets.shift(-1)
        
        # Remove last row since we don't have next week's data for it
        features = features.iloc[:-1]
        targets = targets.iloc[:-1]
        
        # Fill any remaining NaN values with 0
        features = features.fillna(0)
        targets = targets.fillna(0)
        
        print("\nFinal data shapes:")
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        
        print("\nSample features:")
        print(features.head())
        
        print("\nSample targets:")
        print(targets.head())
        
        # Verify no NaN values
        print("\nMissing values in features:")
        print(features.isna().sum())
        
        print("\nMissing values in targets:")
        print(targets.isna().sum())
        
        return features, targets
        
    except Exception as e:
        print(f"Error in create_features_and_target: {str(e)}")
        raise

def train_and_predict(X, y, X_next=None):
    """Train model and make predictions"""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print("\nTraining data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Normalize predictions to sum to 1
        y_pred_train = y_pred_train / y_pred_train.sum(axis=1, keepdims=True)
        y_pred_test = y_pred_test / y_pred_test.sum(axis=1, keepdims=True)
        
        # Calculate R² scores
        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)
        
        print("\nModel Performance:")
        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Testing R² Score: {test_r2:.4f}")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(model.coef_).mean(axis=0)
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print(feature_importance.head(5))
        
        print("\nTest Set Predictions vs Actual:")
        for i, col in enumerate(y.columns):
            print(f"\n{col}:")
            print(f"Predicted: {y_pred_test[0][i]:.4f}")
            print(f"Actual: {y_test.iloc[0][col]:.4f}")
            print(f"Difference: {abs(y_pred_test[0][i] - y_test.iloc[0][col]):.4f}")
        
        # Make predictions for next week if X_next is provided
        if X_next is not None:
            X_next_scaled = scaler.transform(X_next)
            next_pred = model.predict(X_next_scaled)
            
            # Normalize predictions to sum to 1
            next_pred = next_pred / next_pred.sum(axis=1, keepdims=True)
            
            print("\nPredictions for April 18:")
            predictions = pd.Series(next_pred[0], index=y.columns)
            
            # Ensure predictions are non-negative and sum to 1
            predictions = predictions.clip(lower=0)
            predictions = predictions / predictions.sum()
            
            print("\nTop 5 Highest Predicted Prices:")
            print(predictions.nlargest(5))
            
            print("\nBottom 5 Lowest Predicted Prices:")
            print(predictions.nsmallest(5))
            
            # Save predictions to CSV
            predictions.to_csv('predictions.csv')
            print("\nPredictions saved to predictions.csv")
            
            return model, scaler, predictions
        
        return model, scaler
        
    except Exception as e:
        print(f"Error in train_and_predict: {str(e)}")
        raise

def predict_april_18(last_week, model, scaler, valid_columns):
    """Make predictions for April 18"""
    try:
        print("\nCalculating trends for prediction...")
        
        # Create features for prediction
        X_next = pd.DataFrame()
        X_next['Week'] = [last_week['Week'].iloc[0]]
        X_next['vader_mean'] = [0]  # No sentiment data for future week
        
        # Calculate trends for each valid price column
        for col in valid_columns:
            if col != 'Week':
                trend_col = f'trend_{col}'
                trend = calculate_trend(last_week, col).iloc[-1]
                X_next[trend_col] = [trend]
        
        print("\nMaking predictions...")
        # Scale features
        X_next_scaled = scaler.transform(X_next)
        next_pred = model.predict(X_next_scaled)
        
        # Create predictions series (excluding timestamp columns)
        predictions = pd.Series(next_pred[0], index=valid_columns)
        
        # Ensure predictions are non-negative and sum to 1
        predictions = predictions.clip(lower=0)
        predictions = predictions / predictions.sum()
        
        # Format predictions for readability
        print("\nPredicted Price Ranges for April 18:")
        print("-----------------------------------")
        for price_range, value in predictions.nlargest(10).items():
            print(f"{price_range}: {value:.4f}")
        
        print("\nLowest Predicted Price Ranges:")
        print("-----------------------------")
        for price_range, value in predictions.nsmallest(5).items():
            print(f"{price_range}: {value:.4f}")
        
        # Save predictions to CSV (excluding timestamp columns)
        predictions.to_csv('predictions.csv')
        print("\nPredictions saved to predictions.csv")
        
        return predictions
        
    except Exception as e:
        print(f"Error in predict_april_18: {str(e)}")
        raise

def main():
    try:
        # Load and prepare data
        market_data, sentiment_data, valid_columns = load_data()
        
        # Create features and target variables
        X, y = create_features_and_target(market_data, sentiment_data, valid_columns)
        
        # Train model and get predictions
        model, scaler = train_and_predict(X, y)
        
        # Get last week's data for prediction
        last_week = market_data.tail(1)
        
        # Make predictions for April 18
        predictions = predict_april_18(last_week, model, scaler, valid_columns)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 