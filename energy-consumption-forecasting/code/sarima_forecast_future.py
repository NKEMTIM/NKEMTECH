import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# Paths
data_dir = r"c:\Users\DELL\Downloads\enegy consumption\data"
input_csv = os.path.join(data_dir, "clean_merged_energy_usage.csv")
output_csv = os.path.join(data_dir, "sarima_future_forecast.csv")
output_png = os.path.join(data_dir, "sarima_future_forecast_graph.png")

# Fonts for Plot
plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("1. Loading Data...")
df = pd.read_csv(input_csv, parse_dates=['Date'])
ts = df.groupby('Date')['Consumption_MWh'].sum().reset_index()
ts = ts.sort_values('Date').set_index('Date')
ts.index.freq = 'MS' # Month Start frequency

last_date = ts.index.max()
print(f"Target: Consumption_MWh")
print(f"Frequency: Monthly")
print(f"Last Available Date: {last_date.date()}")
print(f"Total historical months: {len(ts)}")

print("\n2. Fitting SARIMA on Full Historical Data...")
# We use the same parameters that performed excellently in the test set.
model = sm.tsa.statespace.SARIMAX(
    ts['Consumption_MWh'],
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)

print("\n3. Forecasting Next 12 Months...")
steps = 12
forecast = results.get_forecast(steps=steps)
pred_mean = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05) # 95% confidence interval

# Ensure index is correctly set to dates
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
pred_mean.index = future_dates
conf_int.index = future_dates

# Build Forecast DataFrame
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Usage': pred_mean,
    'Lower_CI': conf_int.iloc[:, 0],
    'Upper_CI': conf_int.iloc[:, 1]
}).reset_index(drop=True)

forecast_df.to_csv(output_csv, index=False)
print(f"Forecast saved to {output_csv}")

max_pred = forecast_df['Predicted_Usage'].max()
min_pred = forecast_df['Predicted_Usage'].min()
print(f"Highest Predicted: {max_pred}")
print(f"Lowest Predicted: {min_pred}")

print("\n4. Generating Forecast Plot...")
plt.figure(figsize=(14, 7))

# Plot historical actuals
plt.plot(ts.index, ts['Consumption_MWh'], label='Historical Actuals', color='black', linewidth=2)

# Plot future forecast
plt.plot(pred_mean.index, pred_mean, label='SARIMA Future Forecast', color='orange', linestyle='--', linewidth=2.5)

# Plot confidence intervals
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.2, label='95% Confidence Interval')

# Vertical line separating history from future
plt.axvline(x=ts.index[-1], color='red', linestyle=':', linewidth=2, label='Forecast Start Vector')

plt.title('SARIMA Future Forecasting: Energy Consumption (Next 12 Months)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Consumption (MWh)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig(output_png)
print(f"Forecast graph saved to {output_png}")
print("Tasks successfully completed.")
