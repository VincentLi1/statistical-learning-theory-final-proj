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
plt.style.use('default')
sns.set_theme(style="whitegrid")

def load_predictions(model_name):
    """Load predictions CSV file for a given model"""
    try:
        return pd.read_csv(f"{model_name}_pmf_predictions.csv")
    except FileNotFoundError:
        print(f"Warning: No predictions found for {model_name}")
        return None

# Load all available predictions
models = ['lin_reg', 'poly_ridge', 'arima', 'lstm', 'transformer']
model_names = {
    'lin_reg': 'Linear Regression',
    'poly_ridge': 'Polynomial Ridge',
    'arima': 'ARIMA',
    'lstm': 'LSTM',
    'transformer': 'Transformer'
}
predictions = {
    model_names[model]: load_predictions(model)
    for model in models
    if load_predictions(model) is not None
}

if not predictions:
    raise ValueError("No prediction files found!")

# Get bin labels from first available prediction file
bins = next(iter(predictions.values())).columns

# Create figure directory
Path("visualizations").mkdir(exist_ok=True)

# 1. Plot average distributions
plt.figure(figsize=(15, 8))
for model, preds in predictions.items():
    plt.plot(range(len(bins)), preds.mean(), label=model, marker='o', alpha=0.7)
plt.xticks(range(len(bins))[::2], bins[::2], rotation=45, ha='right')
plt.xlabel('Tweet Count Bins')
plt.ylabel('Average Probability')
plt.title('Average Predicted Probability Distribution by Model')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/avg_distributions.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. Box plots for selected bins
# Select every 4th bin to avoid overcrowding
selected_bins = bins[::4]
data_for_box = []
for model, preds in predictions.items():
    for bin_name in selected_bins:
        data_for_box.extend([
            {'Model': model, 'Bin': bin_name, 'Probability': p}
            for p in preds[bin_name]
        ])
df_box = pd.DataFrame(data_for_box)

plt.figure(figsize=(15, 8))
sns.boxplot(data=df_box, x='Bin', y='Probability', hue='Model')
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Predicted Probabilities by Model')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('visualizations/probability_distributions.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. Heatmaps for each model
for model, preds in predictions.items():
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        preds.iloc[:min(50, len(preds))],  # Show first 50 predictions
        cmap='YlOrRd',
        cbar_kws={'label': 'Probability'},
        xticklabels=2  # Show every 2nd label
    )
    plt.xlabel('Tweet Count Bins')
    plt.ylabel('Sample Index')
    plt.title(f'Prediction Heatmap - {model}')
    plt.xticks(range(len(bins))[::2], bins[::2], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'visualizations/heatmap_{model.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()

print("Visualizations have been saved to the 'visualizations' directory:")
print("1. avg_distributions.png - Average predicted distribution for each model")
print("2. probability_distributions.png - Box plots of probability distributions")
print("3. heatmap_*.png - Heatmaps showing individual predictions for each model") 