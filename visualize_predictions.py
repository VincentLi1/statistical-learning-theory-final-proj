"""Visualize predictions from all models.

This script creates several visualizations:
1. Average predicted distribution for each model
2. Box plots showing the distribution of probabilities for each bin
3. Heatmaps showing how predictions vary across samples
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_predictions(model_name):
    """Load predictions from CSV file"""
    df = pd.read_csv(f"{model_name}_pmf_predictions.csv")
    return df

def plot_daily_predictions():
    """Plot daily predictions for all models"""
    models = ['lin_reg', 'poly_ridge', 'arima', 'lstm', 'transformer']
    model_names = ['Linear', 'Polynomial', 'ARIMA', 'LSTM', 'Transformer']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        try:
            df = load_predictions(model)
            # Get the last prediction (most recent)
            pred = df.iloc[-1]
            
            # Find the last meaningful bin (where prob > 0.001) plus buffer
            meaningful_bins = np.where(pred.values > 0.001)[0]
            max_bin = min(meaningful_bins[-1] + 2, len(pred))
            
            # Create bar plot with limited x range
            ax = axes[i]
            bars = ax.bar(range(max_bin), pred.values[:max_bin], alpha=0.7)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0.1:  # Only label significant probabilities
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')
            
            ax.set_title(f'{name} Daily Predictions')
            ax.set_xlabel('Tweet Count Bin')
            ax.set_ylabel('Probability')
            ax.set_xticks(range(max_bin))
            ax.set_xticklabels(pred.index[:max_bin], rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add entropy value in title
            entropy = -np.sum(pred * np.log(pred + 1e-10))
            ax.set_title(f'{name} Daily Predictions\n(Entropy: {entropy:.2f})')
            
        except Exception as e:
            print(f"Error plotting {name}: {str(e)}")
            axes[i].set_title(f'{name} - No Data')
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('new_visualizations/daily_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_weekly_predictions():
    """Plot weekly predictions for all models"""
    models = ['lin_reg', 'poly_ridge', 'arima', 'lstm', 'transformer']
    model_names = ['Linear', 'Polynomial', 'ARIMA', 'LSTM', 'Transformer']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        try:
            df = load_predictions(model)
            # Get the last prediction (most recent)
            pred = df.iloc[-1]
            
            # Find the last meaningful bin (where prob > 0.001) plus buffer
            meaningful_bins = np.where(pred.values > 0.001)[0]
            max_bin = min(meaningful_bins[-1] + 2, len(pred))
            
            # Create bar plot with limited x range
            ax = axes[i]
            bars = ax.bar(range(max_bin), pred.values[:max_bin], alpha=0.7)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0.1:  # Only label significant probabilities
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')
            
            ax.set_title(f'{name} Weekly Predictions')
            ax.set_xlabel('Tweet Count Bin')
            ax.set_ylabel('Probability')
            
            # Scale x-axis labels to show actual tweet counts
            bin_width = 25
            current_tweets = 115  # Current week's tweet count
            x_labels = [f'{current_tweets + j*bin_width}-{current_tweets + (j+1)*bin_width}' 
                       for j in range(max_bin)]
            ax.set_xticks(range(max_bin))
            ax.set_xticklabels(x_labels, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add entropy value in title
            entropy = -np.sum(pred * np.log(pred + 1e-10))
            ax.set_title(f'{name} Weekly Predictions\n(Entropy: {entropy:.2f})')
            
            # Add current week tweets as a vertical line
            current_bin = 0  # First bin starts at current_tweets
            ax.axvline(x=current_bin, color='r', linestyle='--', alpha=0.5)
            ax.text(current_bin, ax.get_ylim()[1]*0.9, 
                   f'Current: {current_tweets}',
                   rotation=90, va='top')
            
        except Exception as e:
            print(f"Error plotting {name}: {str(e)}")
            axes[i].set_title(f'{name} - No Data')
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('new_visualizations/weekly_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison():
    """Plot comparison of all models' predictions"""
    models = ['lin_reg', 'poly_ridge', 'arima', 'lstm', 'transformer']
    model_names = ['Linear', 'Polynomial', 'ARIMA', 'LSTM', 'Transformer']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Find maximum meaningful bin across all models
    max_bin = 0
    for model in models:
        try:
            df = load_predictions(model)
            pred = df.iloc[-1]
            meaningful_bins = np.where(pred.values > 0.001)[0]
            max_bin = max(max_bin, meaningful_bins[-1] + 2)
        except Exception:
            continue
    
    max_bin = min(max_bin, len(pred))
    
    # Daily predictions
    for model, name in zip(models, model_names):
        try:
            df = load_predictions(model)
            pred = df.iloc[-1]
            ax1.plot(range(max_bin), pred.values[:max_bin], label=name, marker='o')
        except Exception as e:
            print(f"Error plotting {name}: {str(e)}")
    
    ax1.set_title('Daily Predictions Comparison')
    ax1.set_xlabel('Tweet Count Bin')
    ax1.set_ylabel('Probability')
    ax1.set_xticks(range(max_bin))
    ax1.set_xticklabels(pred.index[:max_bin], rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Weekly predictions with scaled x-axis
    bin_width = 25
    current_tweets = 115  # Current week's tweet count
    x_labels = [f'{current_tweets + j*bin_width}-{current_tweets + (j+1)*bin_width}' 
               for j in range(max_bin)]
    
    for model, name in zip(models, model_names):
        try:
            df = load_predictions(model)
            pred = df.iloc[-1]
            ax2.plot(range(max_bin), pred.values[:max_bin], label=name, marker='o')
        except Exception as e:
            print(f"Error plotting {name}: {str(e)}")
    
    ax2.set_title('Weekly Predictions Comparison')
    ax2.set_xlabel('Tweet Count Bin')
    ax2.set_ylabel('Probability')
    ax2.set_xticks(range(max_bin))
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('new_visualizations/predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_entropy_comparison():
    """Plot entropy comparison across models"""
    models = ['lin_reg', 'poly_ridge', 'arima', 'lstm', 'transformer']
    model_names = ['Linear', 'Polynomial', 'ARIMA', 'LSTM', 'Transformer']
    
    entropies = []
    for model in models:
        try:
            df = load_predictions(model)
            pred = df.iloc[-1]
            entropy = -np.sum(pred * np.log(pred + 1e-10))
            entropies.append(entropy)
        except Exception as e:
            print(f"Error calculating entropy for {model}: {str(e)}")
            entropies.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(model_names, entropies, alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    ax.set_title('Model Entropy Comparison')
    ax.set_xlabel('Model')
    ax.set_ylabel('Entropy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('new_visualizations/entropy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all visualizations"""
    print("Generating visualizations...")
    plot_daily_predictions()
    plot_weekly_predictions()
    plot_comparison()
    plot_entropy_comparison()
    print("Visualizations saved as PNG files.")

if __name__ == "__main__":
    main() 