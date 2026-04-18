import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

def run_bilstm():
    print("--- Bidirectional LSTM (Bi-LSTM) Baseline ---")
    
    # 1. Load dataset cleanly
    df = pd.read_csv('../data/cleaned_merged.csv')
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 2. Sort by datetime and confirm interval
    df = df.sort_values('datetime').reset_index(drop=True)
    time_diff = df['datetime'].diff().mode()[0]
    print(f"Sampling interval confirmed: {time_diff}")
    
    # 3. Select the same feature columns
    features = ['env_slopesolar', 'env_modtemp']
    target = 'pow_acp'
    cols = ['datetime', target] + features
    
    # 4. Drop non-selected NAs
    df_clean = df[cols].dropna().reset_index(drop=True)
    
    # 10. Memory safe
    MAX_ROWS = 20000
    if len(df_clean) > MAX_ROWS:
        df_clean = df_clean.tail(MAX_ROWS).reset_index(drop=True)
        
    n = len(df_clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    print(f"\nDataset Strategy: {n} Total. Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}")
    
    # 7. Split in time order
    train_df = df_clean.iloc[:train_end]
    val_df = df_clean.iloc[train_end:val_end]
    test_df = df_clean.iloc[val_end:]
    
    # 8. Scale input features (training only)
    feature_cols = [target] + features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feature_cols].values)
    
    scaled_train = scaler.transform(train_df[feature_cols].values)
    scaled_val = scaler.transform(val_df[feature_cols].values)
    scaled_test = scaler.transform(test_df[feature_cols].values)
    
    # 5 & 6. Create sequences
    seq_length = 12
    
    def create_sequences(data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            x = data[i:(i + seq_len)]
            y = data[i + seq_len, 0] # Next immediate step (t+1)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
        
    X_train, y_train = create_sequences(scaled_train, seq_length)
    X_val, y_val = create_sequences(scaled_val, seq_length)
    X_test, y_test = create_sequences(scaled_test, seq_length)
    
    print(f"Sequence Shape: {X_train.shape}")
    
    # 9 & 10. Build lightweight Bi-LSTM
    print("\nBuilding Bi-LSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = Sequential()
    # Using 16 units to match the ~32 parameter complexity of the Uni-LSTM
    model.add(Bidirectional(LSTM(16, activation='tanh'), input_shape=input_shape))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    # 11. Train with lightweight settings
    print(model.summary())
    print("\nTraining...")
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 12. Evaluate
    print("\nEvaluating Bi-LSTM on Test Set...")
    test_preds = model.predict(X_test, verbose=0)
    
    dummy_pred = np.zeros((len(test_preds), len(feature_cols)))
    dummy_pred[:, 0] = test_preds[:, 0]
    real_preds = scaler.inverse_transform(dummy_pred)[:, 0]
    
    dummy_actual = np.zeros((len(y_test), len(feature_cols)))
    dummy_actual[:, 0] = y_test
    real_actuals = scaler.inverse_transform(dummy_actual)[:, 0]
    
    rmse = np.sqrt(mean_squared_error(real_actuals, real_preds))
    mae = mean_absolute_error(real_actuals, real_preds)
    r2 = r2_score(real_actuals, real_preds)
    
    print(f"Bi-LSTM R2:   {r2:.4f}")
    print(f"Bi-LSTM RMSE: {rmse:.2f} W")
    print(f"Bi-LSTM MAE:  {mae:.2f} W")
    
    # Output to text file
    with open('bilstm_metrics.txt', 'w') as f:
        f.write("--- Bi-LSTM Final Results ---\n")
        f.write(f"R2: {r2:.4f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\n")

    # 13 & 14. Plots
    print("\nSaving Plots...")
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Bi-LSTM Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Scaled)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bilstm_loss.png', dpi=150)
    plt.close()
    
    window_sz = 200
    plt.figure(figsize=(12, 6))
    plt.plot(real_actuals[:window_sz], label='Actual AC Power', color='black', alpha=0.6, linewidth=2)
    plt.plot(real_preds[:window_sz], label='Bi-LSTM Predicted Power (t+1)', color='tab:green', linestyle='--', linewidth=2)
    plt.title('Single-Step Forecast Overlay: Bi-LSTM Actual vs Predicted', fontsize=14)
    plt.xlabel('Time Steps (10-minute intervals)')
    plt.ylabel('AC Power (W)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('bilstm_pred.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    run_bilstm()
