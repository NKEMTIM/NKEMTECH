import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. Load the cleaned merged dataset & setup
# ==========================================
print("1. Loading dataset...")
df = pd.read_csv('../data/cleaned_merged.csv')
if 'Unnamed: 0' in df.columns:
    df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'])

# 2. Sort by datetime
df = df.sort_values('datetime').reset_index(drop=True)

# 4. First print the dataset shape and sampling interval.
print(f"Original dataset shape: {df.shape}")
time_diff = df['datetime'].diff().mode()[0]
print(f"Sampling interval detected: {time_diff}")

# 3. Use only a small set of reliable columns
features = ['env_slopesolar', 'env_modtemp']
target = 'pow_acp'
columns_to_keep = ['datetime', target] + features

# 5. Drop rows with missing values only in the selected columns.
df_clean = df[columns_to_keep].dropna().reset_index(drop=True)
print(f"Dataset shape after dropping NaNs in selected columns: {df_clean.shape}")

# Use a smaller continuous subset if requested to prevent memory hang
MAX_ROWS = 20000
if len(df_clean) > MAX_ROWS:
    print(f"Warning: Dataset is large ({len(df_clean)} rows). Reducing to last {MAX_ROWS} rows for memory safety baseline.")
    df_clean = df_clean.tail(MAX_ROWS).reset_index(drop=True)

# ==========================================
# Split Data (70% Train, 15% Val, 15% Test) in time order
# ==========================================
print("\n8. Splitting dataset in time order...")
n = len(df_clean)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df_clean.iloc[:train_end]
val_df = df_clean.iloc[train_end:val_end]
test_df = df_clean.iloc[val_end:]

print(f"Train rows: {len(train_df)}")
print(f"Val rows: {len(val_df)}")
print(f"Test rows: {len(test_df)}")

# ==========================================
# 9. Scale features using training data only
# ==========================================
print("\n9. Scaling features...")
feature_cols = [target] + features  # index 0 is target
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit only on training to prevent data leakage
scaler.fit(train_df[feature_cols].values)

scaled_train = scaler.transform(train_df[feature_cols].values)
scaled_val = scaler.transform(val_df[feature_cols].values)
scaled_test = scaler.transform(test_df[feature_cols].values)

# ==========================================
# 6 & 7. Create sequences for single-step forecasting
# ==========================================
seq_length = 12

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)] # Past 12 steps
        y = data[i + seq_len, 0]  # Next immediate step of pow_acp (index 0)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

print(f"\n7. Creating sequences with length = {seq_length}...")
X_train, y_train = create_sequences(scaled_train, seq_length)
X_val, y_val = create_sequences(scaled_val, seq_length)
X_test, y_test = create_sequences(scaled_test, seq_length)

print(f"Train sequence shape: {X_train.shape}")
print(f"Val sequence shape: {X_val.shape}")
print(f"Test sequence shape: {X_test.shape}")

# ==========================================
# 10. Build a very small LSTM model
# ==========================================
print("\n10. Building Unidirectional LSTM model...")
input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential()
# One LSTM layer, very small hidden size for baseline speed
model.add(LSTM(32, input_shape=input_shape, activation='tanh'))
# One Dense output layer for single-step prediction
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# ==========================================
# 11. Train with lightweight settings
# ==========================================
print("\n11. Training Model...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 12. Evaluate Model
# ==========================================
print("\n12. Evaluating on Test Set...")
test_preds = model.predict(X_test)

# Inverse transform to get real watt values
# We create dummy arrays matching the original feature count to perform strict inverse transform
dummy_pred = np.zeros((len(test_preds), len(feature_cols)))
dummy_pred[:, 0] = test_preds[:, 0]
real_preds = scaler.inverse_transform(dummy_pred)[:, 0]

dummy_actual = np.zeros((len(y_test), len(feature_cols)))
dummy_actual[:, 0] = y_test
real_actuals = scaler.inverse_transform(dummy_actual)[:, 0]

rmse = np.sqrt(mean_squared_error(real_actuals, real_preds))
mae = mean_absolute_error(real_actuals, real_preds)
r2 = r2_score(real_actuals, real_preds)

print(f"Test RMSE: {rmse:.2f} W")
print(f"Test MAE:  {mae:.2f} W")
print(f"Test R²:   {r2:.4f}")

# ==========================================
# 13. Plotting
# ==========================================
print("\n13. Generating plots...")
# Plot 1: Training vs Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Baseline LSTM Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (Scaled)')
plt.legend()
plt.tight_layout()
plt.savefig('baseline_lstm_loss.png', dpi=150)
plt.close()

# Plot 2: Actual vs Predicted (Small test window to actually see the lines)
window_size = 200 # approx 1.5 days of 10-minute intervals
plt.figure(figsize=(12, 6))
plt.plot(real_actuals[:window_size], label='Actual AC Power', color='tab:blue', alpha=0.8, linewidth=2)
plt.plot(real_preds[:window_size], label='Predicted AC Power (t+1)', color='tab:orange', linestyle='--', linewidth=2)
plt.title('Baseline Single-Step LSTM: Actual vs Predicted (200 step window)', fontsize=14, pad=15)
plt.xlabel('Time Steps (10-minute intervals)')
plt.ylabel('AC Power (W)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('baseline_lstm_pred.png', dpi=150)
plt.close()

print("\nModel complete. Plots saved as baseline_lstm_loss.png and baseline_lstm_pred.png.")
