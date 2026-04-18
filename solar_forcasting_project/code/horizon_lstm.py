import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def run_multi_horizon():
    print("--- Multi-Horizon Baseline LSTM Evaluation (1 to 6 steps ahead) ---")
    
    # 1. Load dataset cleanly
    df = pd.read_csv('../data/cleaned_merged.csv')
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Subset to keep execution fast
    features = ['env_slopesolar', 'env_modtemp']
    target = 'pow_acp'
    cols = ['datetime', target] + features
    df_clean = df[cols].dropna().reset_index(drop=True)
    
    MAX_ROWS = 20000
    if len(df_clean) > MAX_ROWS:
        df_clean = df_clean.tail(MAX_ROWS).reset_index(drop=True)
        
    n = len(df_clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    print(f"Dataset Size: {n} rows. (Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end})")
    
    train_df = df_clean.iloc[:train_end]
    val_df = df_clean.iloc[train_end:val_end]
    test_df = df_clean.iloc[val_end:]
    
    feature_cols = [target] + features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feature_cols].values)
    
    scaled_train = scaler.transform(train_df[feature_cols].values)
    scaled_val = scaler.transform(val_df[feature_cols].values)
    scaled_test = scaler.transform(test_df[feature_cols].values)
    
    # Adjustable sequence creator for horizons
    def create_sequences(data, seq_len, horizon):
        xs, ys = [], []
        # Need to ensure we don't index out of bounds
        for i in range(len(data) - seq_len - horizon + 1):
            x = data[i:(i + seq_len)]
            y = data[i + seq_len + horizon - 1, 0] # target column is index 0
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
        
    seq_length = 12
    horizons = [1, 2, 3, 4, 5, 6]
    
    results = {
        'horizon': [],
        'r2': [],
        'rmse': [],
        'mae': []
    }
    
    # Store predictions for the plotting phase
    test_actuals_dict = {}
    test_preds_dict = {}
    
    print("\nStarting horizon loop...")
    for h in horizons:
        print(f"\n==============================================")
        print(f"Training Model for Horizon = {h} (t+{h})")
        
        X_train, y_train = create_sequences(scaled_train, seq_length, h)
        X_val, y_val = create_sequences(scaled_val, seq_length, h)
        X_test, y_test = create_sequences(scaled_test, seq_length, h)
        
        # Extremely lightweight model since we train 6 times
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = Sequential()
        model.add(LSTM(16, input_shape=input_shape, activation='tanh'))  # Reduced to 16 for speed
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=0)
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=64, # Increased for speed
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
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
        
        print(f"Horizon {h} | R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        
        results['horizon'].append(h)
        results['r2'].append(r2)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        
        test_actuals_dict[h] = real_actuals
        test_preds_dict[h] = real_preds

    # Summary Table
    df_res = pd.DataFrame(results)
    print("\n--- Summary Table of Horizons ---")
    print(df_res.to_string(index=False))
    
    with open('horizon_summary.txt', 'w') as f:
        f.write(df_res.to_string(index=False))

    # --- Plotting ---
    print("\nGenerating Plots...")
    
    # 1. R2 Plot
    plt.figure(figsize=(7, 4))
    plt.plot(df_res['horizon'], df_res['r2'], marker='o', color='green', linewidth=2)
    plt.title('Prediction Horizon vs R² Score')
    plt.xlabel('Horizon (10-min steps ahead)')
    plt.ylabel('R² Score')
    plt.grid(True, linestyle='--')
    plt.xticks(horizons)
    plt.tight_layout()
    plt.savefig('horizon_r2.png', dpi=150)
    plt.close()
    
    # 2. RMSE Plot
    plt.figure(figsize=(7, 4))
    plt.plot(df_res['horizon'], df_res['rmse'], marker='o', color='red', linewidth=2)
    plt.title('Prediction Horizon vs RMSE')
    plt.xlabel('Horizon (10-min steps ahead)')
    plt.ylabel('RMSE (W)')
    plt.grid(True, linestyle='--')
    plt.xticks(horizons)
    plt.tight_layout()
    plt.savefig('horizon_rmse.png', dpi=150)
    plt.close()
    
    # 3. Illustrative Forecast Plot Overlay
    # To properly overlay, all horizon predictions must be shifted by their horizon 
    # relative to a unified 'Actuals' baseline on the x-axis.
    # We take the actuals from horizon 1 as the continuous baseline.
    window_sz = 150
    plt.figure(figsize=(14, 6))
    
    # Since H=1 predicts t+1, H=6 predicts t+6 from the same input window.
    # A single input window starting at index 0 ends at seq_len. 
    # H=1 predicts index seq_len+0. H=6 predicts index seq_len+5.
    # Therefore, we can plot test_actuals_dict[1] starting from x=0.
    # test_preds_dict[1] starts at x=0
    # test_preds_dict[3] starts at x=2
    # test_preds_dict[6] starts at x=5
    
    plt.plot(test_actuals_dict[1][:window_sz], label='Actual AC Power', color='black', linewidth=2, alpha=0.6)
    
    # Plotting h=1
    plt.plot(np.arange(0, window_sz), test_preds_dict[1][:window_sz], label='H=1 (t+10m) Prediction', linewidth=1.5)
    # Plotting h=3
    plt.plot(np.arange(2, window_sz+2), test_preds_dict[3][:window_sz], label='H=3 (t+30m) Prediction', linewidth=1.5, linestyle='--')
    # Plotting h=6
    plt.plot(np.arange(5, window_sz+5), test_preds_dict[6][:window_sz], label='H=6 (t+60m) Prediction', linewidth=1.5, linestyle=':')
    
    plt.title('Illustrative Multi-Horizon Forecast: Increasing Uncertainty over Time', fontsize=14)
    plt.xlabel('Time Steps (10-minute intervals)')
    plt.ylabel('AC Power (W)')
    plt.xlim(0, window_sz+5)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig('horizon_forecast_demo.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress tf logs
    run_multi_horizon()
