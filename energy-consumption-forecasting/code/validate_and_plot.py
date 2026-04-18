import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    print("--- Part A: Validation ---")
    # 1 & 2 & 3. Load the actual vs predicted results and metrics
    metrics = pd.read_csv('../data/model_comparison_results.csv')
    print("\nModel Comparison Table:")
    print(metrics.to_string(index=False))
    
    # Save as validation_results.csv
    metrics.to_csv('../data/validation_results.csv', index=False)
    
    # Load predictions
    sarima_preds = pd.read_csv('../data/sarima_test_predictions.csv')
    lstm_preds = pd.read_csv('../data/lstm_test_predictions.csv')
    
    sarima_preds['Date'] = pd.to_datetime(sarima_preds['Date'])
    lstm_preds['Date'] = pd.to_datetime(lstm_preds['Date'])
    
    sarima_preds.set_index('Date', inplace=True)
    lstm_preds.set_index('Date', inplace=True)
    
    # Calculate residuals
    sarima_preds['Residual'] = sarima_preds['Actual'] - sarima_preds['Predicted_SARIMA']
    lstm_preds['Residual'] = lstm_preds['Actual'] - lstm_preds['Predicted_LSTM']
    
    # 6. Create 2 simple validation visuals
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Actual vs Predicted line plot
    axes[0].plot(sarima_preds.index, sarima_preds['Actual'], label='Actual', color='black', linewidth=2)
    axes[0].plot(sarima_preds.index, sarima_preds['Predicted_SARIMA'], label='SARIMA', linestyle='--')
    axes[0].plot(lstm_preds.index, lstm_preds['Predicted_LSTM'], label='LSTM', linestyle='-.')
    axes[0].set_title('Actual vs Predicted Energy Usage')
    axes[0].set_ylabel('Energy Usage')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # Residual Histogram
    sns.histplot(sarima_preds['Residual'], bins=10, color='blue', alpha=0.5, label='SARIMA', ax=axes[1], kde=True)
    sns.histplot(lstm_preds['Residual'], bins=10, color='red', alpha=0.5, label='LSTM', ax=axes[1], kde=True)
    axes[1].set_title('Residuals Distribution (Errors)')
    axes[1].set_xlabel('Error (Actual - Predicted)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('../data/validation_plot.png')
    print("\nSaved validation plot to data/validation_plot.png")
    
if __name__ == "__main__":
    main()
