import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# Paths
data_dir = r"c:\Users\DELL\Downloads\enegy consumption\data"
input_csv = os.path.join(data_dir, "clean_merged_energy_usage.csv")
output_csv = os.path.join(data_dir, "future_forecast_4years_combined.csv")
output_png = os.path.join(data_dir, "future_forecast_4years_combined_graph.png")

# Fonts for Plot
plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("1. Loading Data...")
df = pd.read_csv(input_csv, parse_dates=['Date'])
ts = df.groupby('Date')['Consumption_MWh'].sum().reset_index()
ts = ts.sort_values('Date').set_index('Date')
ts.index.freq = 'MS'

last_date = ts.index.max()
forecast_steps = 48 # 4 years

print(f"Target: Consumption_MWh")
print(f"Frequency: Monthly")
print(f"Last Available Date: {last_date.date()}")
print(f"Total historical months: {len(ts)}")
print(f"Forecast Horizon: {forecast_steps} months (4 years)")

print("\n2. Fitting SARIMA on Full Historical Data...")
sarima_model = sm.tsa.statespace.SARIMAX(
    ts['Consumption_MWh'],
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_results = sarima_model.fit(disp=False)

print("\n3. Forecasting SARIMA for Next 4 Years...")
forecast = sarima_results.get_forecast(steps=forecast_steps)
sarima_pred_mean = forecast.predicted_mean
sarima_conf_int = forecast.conf_int(alpha=0.05) # 95% CI

future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=forecast_steps, freq='MS')
sarima_pred_mean.index = future_dates
sarima_conf_int.index = future_dates


print("\n4. Preparing for LSTM...")
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

print("\n5. Training LSTM on Full History...")
tf.random.set_seed(42)
np.random.seed(42)

lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(time_steps, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.fit(X_train, y_train, epochs=150, batch_size=8, verbose=0)

print("\n6. Forecasting LSTM for Next 4 Years iteratively...")
current_input = ts_scaled[-time_steps:].reshape((1, time_steps, 1))
lstm_predictions_scaled = []

for _ in range(forecast_steps):
    pred = lstm_model.predict(current_input, verbose=0)[0]
    lstm_predictions_scaled.append(pred)
    current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)
    
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled).flatten()

print("\n7. Creating Combined Forecast Table...")
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'SARIMA_Forecast': sarima_pred_mean.values,
    'SARIMA_Lower_CI': sarima_conf_int.iloc[:, 0].values,
    'SARIMA_Upper_CI': sarima_conf_int.iloc[:, 1].values,
    'LSTM_Forecast': lstm_predictions
})

forecast_df.to_csv(output_csv, index=False)
print(f"Forecast saved to {output_csv}")

print("\n8. Generating Combined Forecast Plot...")
plt.figure(figsize=(16, 8))

# Historical
plt.plot(ts.index, ts['Consumption_MWh'], label='Historical Actuals', color='black', linewidth=2)

# SARIMA
plt.plot(future_dates, sarima_pred_mean, label='SARIMA Forecast', color='orange', linestyle='-', linewidth=2.5)
plt.fill_between(future_dates, sarima_conf_int.iloc[:, 0], sarima_conf_int.iloc[:, 1], color='orange', alpha=0.2, label='SARIMA 95% CI')

# LSTM
plt.plot(future_dates, lstm_predictions, label='LSTM Forecast', color='#3498db', linestyle='--', linewidth=2.5)

# Vertical line
plt.axvline(x=ts.index[-1], color='red', linestyle=':', linewidth=2, label='Forecast Start')

plt.title('SARIMA vs LSTM: 4-Year Energy Consumption Future Forecast', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Consumption (MWh)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()

plt.savefig(output_png)
print(f"Forecast graph saved to {output_png}")

print("\n--- SUMMARY ---")
max_sarima = forecast_df.loc[forecast_df['SARIMA_Forecast'].idxmax()]
min_sarima = forecast_df.loc[forecast_df['SARIMA_Forecast'].idxmin()]
max_lstm = forecast_df.loc[forecast_df['LSTM_Forecast'].idxmax()]
min_lstm = forecast_df.loc[forecast_df['LSTM_Forecast'].idxmin()]

print(f"Forecast Horizon: {forecast_steps} months")
print(f"Date Range Forecasted: {future_dates[0].strftime('%Y-%m')} to {future_dates[-1].strftime('%Y-%m')}")
print(f"SARIMA Highest Predicted Period: {max_sarima['Date'].strftime('%Y-%m')} ({max_sarima['SARIMA_Forecast']:.2f} MWh)")
print(f"LSTM Highest Predicted Period: {max_lstm['Date'].strftime('%Y-%m')} ({max_lstm['LSTM_Forecast']:.2f} MWh)")
print(f"SARIMA Lowest Predicted Period: {min_sarima['Date'].strftime('%Y-%m')} ({min_sarima['SARIMA_Forecast']:.2f} MWh)")
print(f"LSTM Lowest Predicted Period: {min_lstm['Date'].strftime('%Y-%m')} ({min_lstm['LSTM_Forecast']:.2f} MWh)")
print("Tasks successfully completed.")
