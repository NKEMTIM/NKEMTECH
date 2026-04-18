import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import statsmodels.api as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Paths
data_dir = r"c:\Users\DELL\Downloads\enegy consumption\data"
output_md = os.path.join(data_dir, "model_results_summary.md")

plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("1. Loading and Preparing Data...")
# Load baseline clean data and aggregate to macro level for readable univariate time-series plotting
df = pd.read_csv(os.path.join(data_dir, "clean_merged_energy_usage.csv"), parse_dates=['Date'])
ts = df.groupby('Date')['Consumption_MWh'].sum().reset_index()
ts = ts.sort_values('Date').set_index('Date')

print("   - Target Energy Usage Column: 'Consumption_MWh'")
print("   - Frequency is strictly Monthly.")
print(f"   - Date Range: {ts.index.min().strftime('%Y-%m')} to {ts.index.max().strftime('%Y-%m')}")
print(f"   - Missing timestamps or spacing: {ts.isnull().sum().sum()} missing.")

# Split: 2018-2023 (Train), 2024 (Val), 2025 (Test)
train_data = ts[ts.index.year <= 2023]
val_data = ts[ts.index.year == 2024]
test_data = ts[ts.index.year == 2025]

print("\n2. Splitting Dataset...")
print(f"   - Train shape: {train_data.shape} ({train_data.index.min().date()} to {train_data.index.max().date()})")
print(f"   - Val shape:   {val_data.shape} ({val_data.index.min().date()} to {val_data.index.max().date()})")
print(f"   - Test shape:  {test_data.shape} ({test_data.index.min().date()} to {test_data.index.max().date()})")

print("\n3. SARIMA Modeling...")
# Selected order based on previous EDA: flat trend (d=0), 12-month seasonality (m=12)
# Using generic robust starting parameters: ARIMA(1,0,1)(1,1,1,12)
model_sarima = sm.tsa.statespace.SARIMAX(
    train_data['Consumption_MWh'],
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit = model_sarima.fit(disp=False)

# Forecast on Test set using actuals leading up to test
# We use get_prediction to predict the 12 months of test data.
# Note: dynamic=False means it uses the true past values for 1-step ahead
res_sarima = sarima_fit.get_prediction(start=test_data.index[0], end=test_data.index[-1], dynamic=False)
sarima_pred = res_sarima.predicted_mean

print("\n4. LSTM Modeling...")
# Scale using only training
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data[['Consumption_MWh']])
val_scaled = scaler.transform(val_data[['Consumption_MWh']])
test_scaled = scaler.transform(test_data[['Consumption_MWh']])

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 12

# To predict Test, we need the last 12 months leading up to each test point.
# We concatenate Train + Val + Test and then scale
full_series = pd.concat([train_data, val_data, test_data])
full_scaled = scaler.transform(full_series[['Consumption_MWh']])

X_train, y_train = create_sequences(train_scaled, time_steps)
# For validation, we use the end of train + val
val_series = pd.concat([train_data.iloc[-time_steps:], val_data])
val_seq_scaled = scaler.transform(val_series[['Consumption_MWh']])
X_val, y_val = create_sequences(val_seq_scaled, time_steps)

# For test, we use end of val + test
test_series = pd.concat([val_data.iloc[-time_steps:], test_data])
test_seq_scaled = scaler.transform(test_series[['Consumption_MWh']])
X_test, y_test = create_sequences(test_seq_scaled, time_steps)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(time_steps, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_lstm.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_val, y_val), callbacks=[es], verbose=0)

# Predict Test
lstm_pred_scaled = model_lstm.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()

print("\n5. Evaluating Models...")
metrics = []
for model_name, preds in zip(['SARIMA', 'LSTM'], [sarima_pred, lstm_pred]):
    mae = mean_absolute_error(test_data['Consumption_MWh'], preds)
    rmse = np.sqrt(mean_squared_error(test_data['Consumption_MWh'], preds))
    r2 = r2_score(test_data['Consumption_MWh'], preds)
    mape = mean_absolute_percentage_error(test_data['Consumption_MWh'], preds)
    metrics.append({'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape})

metrics_df = pd.DataFrame(metrics).round(4)
print(metrics_df.to_markdown(index=False))

print("\n6. Saving Outputs...")
sarima_df = pd.DataFrame({'Date': test_data.index, 'Actual': test_data['Consumption_MWh'], 'Predicted_SARIMA': sarima_pred.values})
lstm_df = pd.DataFrame({'Date': test_data.index, 'Actual': test_data['Consumption_MWh'], 'Predicted_LSTM': lstm_pred})

sarima_df.to_csv(os.path.join(data_dir, "sarima_test_predictions.csv"), index=False)
lstm_df.to_csv(os.path.join(data_dir, "lstm_test_predictions.csv"), index=False)
metrics_df.to_csv(os.path.join(data_dir, "model_comparison_results.csv"), index=False)

# Prediction Graph
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_data['Consumption_MWh'], label='Actual Usage', marker='o', linewidth=2, color='black')
plt.plot(test_data.index, sarima_pred, label='SARIMA Prediction', marker='x', linestyle='dashed', linewidth=2, color='#e74c3c')
plt.plot(test_data.index, lstm_pred, label='LSTM Prediction', marker='s', linestyle='dotted', linewidth=2, color='#3498db')
plt.title('Time-Series Forecasting: Actual vs SARIMA vs LSTM (Test Set: 2025)', fontsize=16)
plt.xlabel('Date (2025)')
plt.ylabel('Total Consumption (MWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
graph_path = os.path.join(data_dir, "prediction_graph.png")
plt.savefig(graph_path)
plt.close()

with open(output_md, 'w', encoding='utf-8') as f:
    f.write("# Forecasting Output Generated Successfully.\n")
print("All tasks completed successfully. Check output variables for summary.")
