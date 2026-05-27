import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings("ignore")

def main():
    project_dir = Path(r"c:\Users\Administrator\Downloads\Energy Forecasting")
    input_csv = project_dir / "analysis" / "model_inputs" / "energy_features_xgb.csv"
    outputs_dir = project_dir / "analysis" / "model_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = outputs_dir / "xgb_energy_model.pkl"
    plot_actual_pred = outputs_dir / "xgb_actual_vs_predicted.png"
    plot_importance = outputs_dir / "xgb_feature_importance.png"
    report_path = outputs_dir / "xgb_metrics_report.txt"

    print("Loading data...")
    df = pd.read_csv(input_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Sorting and encoding...")
    # 2. Sort by timestamp (Time-based split requires chronologically sorted data)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # 3. Encode building_type
    le = LabelEncoder()
    df['building_type'] = le.fit_transform(df['building_type'])
    
    # 4. Define features and target
    target = 'load_kw'
    drop_cols = ['timestamp', target]
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df[target]
    
    print("Splitting data (Time-based: 70% / 15% / 15%)...")
    # 5. Time-based split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    print("Training XGBoost Regressor...")
    # 6. Train model
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    
    # We evaluate on val set.
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    
    print("Evaluating on test set...")
    # 7. Evaluate
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    report_content = [
        "Step 5: XGBoost Model Evaluation Report",
        "="*50,
        f"Train Size: {len(X_train)}",
        f"Validation Size: {len(X_val)}",
        f"Test Size: {len(X_test)}",
        "\nTest Set Metrics:",
        f"MAE:  {mae:.4f}",
        f"RMSE: {rmse:.4f}",
        f"R²:   {r2:.4f}",
        "\nModel Parameters:",
        str(model.get_params())
    ]
    
    # 10. Save report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
        
    print("Saving model and generating plots...")
    # 9. Save Model
    with open(model_path, "wb") as f:
        pickle.dump({
            'model': model, 
            'label_encoder': le, 
            'features': features,
            'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2}
        }, f)
        
    # 8. Plots
    # Plot Actual vs Predicted (Plotting first 300 test points for clarity)
    plt.figure(figsize=(14, 5))
    plt.plot(y_test.values[:300], label='Actual', marker='.', alpha=0.8)
    plt.plot(preds[:300], label='Predicted', marker='.', alpha=0.8)
    plt.title('Actual vs Predicted Load (kW) - First 300 Test Samples')
    plt.xlabel('Time Step')
    plt.ylabel('Load (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_actual_pred, dpi=300)
    plt.close()
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    idx = np.argsort(importances)
    plt.barh(range(len(idx)), importances[idx], align='center', color='skyblue')
    plt.yticks(range(len(idx)), [features[i] for i in idx])
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(plot_importance, dpi=300)
    plt.close()
    
    print("\n" + "="*50)
    print("Done! Final Evaluation metrics on TEST set:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Model saved to: {model_path}")
    print("="*50)

if __name__ == "__main__":
    main()
