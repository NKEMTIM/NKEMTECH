import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# Paths
data_dir = r"c:\Users\DELL\Downloads\enegy consumption\data"
input_csv = os.path.join(data_dir, "clean_merged_energy_usage.csv")
output_csv = os.path.join(data_dir, "lstm_future_forecast.csv")
output_png = os.path.join(data_dir, "lstm_future_forecast_graph.png")

# Fonts for Plot
plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("1. Loading Data...")
df = pd.read_csv(input_csv, parse_dates=['Date'])
ts = df.groupby('Date')['Consumption_MWh'].sum().reset_index()
ts = ts.sort_values('Date').set_index('Date')
ts.index.freq = 'MS'

last_date = ts.index.max()
print(f"Target: Consumption_MWh")
print(f"Frequency: Monthly")
print(f"Last Available Date: {last_date.date()}")
print(f"Total historical months: {len(ts)}")

print("\n2. Preparing for LSTM...")
scaler = MinMaxScaler(feature_range=(0, 1))
ts_scaled = scaler.fit_transform(ts[['Consumption_MWh']])

time_steps = 12

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(ts_scaled, time_steps)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print("\n3. Training LSTM on Full History...")
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Training slightly longer without validation since we want full convergence on all data
model.fit(X_train, y_train, epochs=150, batch_size=8, verbose=0)

print("\n4. Forecasting Next 12 Months iteratively...")
future_steps = 12
current_input = ts_scaled[-time_steps:].reshape((1, time_steps, 1))
predictions_scaled = []

for _ in range(future_steps):
    pred = model.predict(current_input, verbose=0)[0]
    predictions_scaled.append(pred)
    # Update input sequence: drop first point, append new prediction
    current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)
    
predictions = scaler.inverse_transform(predictions_scaled).flatten()

future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=future_steps, freq='MS')

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Usage': predictions
})

forecast_df.to_csv(output_csv, index=False)
print(f"Forecast saved to {output_csv}")

print("\n5. Generating Forecast Plot...")
plt.figure(figsize=(14, 7))

# Historical
plt.plot(ts.index, ts['Consumption_MWh'], label='Historical Actuals', color='black', linewidth=2)

# Forecast
plt.plot(future_dates, predictions, label='LSTM Future Forecast', color='#3498db', linestyle='--', linewidth=2.5)

# Vertical line
plt.axvline(x=ts.index[-1], color='red', linestyle=':', linewidth=2, label='Forecast Start Vector')

plt.title('LSTM Future Forecasting: Energy Consumption (Next 12 Months)', fontsize=16)
plt.xlabel('Date (2026)', fontsize=12)
plt.ylabel('Total Consumption (MWh)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig(output_png)
print(f"Forecast graph saved to {output_png}")
print("Tasks successfully completed.")
