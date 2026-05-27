import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def main():
    # Paths
    project_dir = Path(r"c:\Users\Administrator\Downloads\Energy Forecasting")
    input_csv = project_dir / "analysis" / "model_inputs" / "energy_modeling_weather_filled.csv"
    output_csv = project_dir / "analysis" / "model_inputs" / "energy_features_xgb.csv"
    report_path = project_dir / "analysis" / "model_inputs" / "step4_feature_report.txt"

    print("Loading data...")
    df = pd.read_csv(input_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 2. Sort by building_type and timestamp
    df = df.sort_values(by=['building_type', 'timestamp']).reset_index(drop=True)

    # 3. Create time features
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    print("Creating lag and rolling features...")
    # 4 & 5. Lag and rolling features per building_type
    # lags
    df['load_lag_1h'] = df.groupby('building_type')['load_kw'].shift(1)
    df['load_lag_24h'] = df.groupby('building_type')['load_kw'].shift(24)
    df['load_lag_168h'] = df.groupby('building_type')['load_kw'].shift(168)
    
    # rolling
    # For rolling in pandas, we need to handle the index properly if it returns a multi-index
    df['load_roll_mean_24h'] = df.groupby('building_type')['load_kw'].transform(lambda x: x.rolling(24).mean())
    df['load_roll_std_24h'] = df.groupby('building_type')['load_kw'].transform(lambda x: x.rolling(24).std())

    # 6. Drop rows with NaN caused by lag/rolling features
    df_clean = df.dropna().reset_index(drop=True)

    print("Saving features...")
    # 7. Save final dataset
    df_clean.to_csv(output_csv, index=False)

    # 8. Save report & Print
    report_content = [
        "Step 4: Feature Engineering Report",
        "="*50,
        f"Final Dataset Shape: {df_clean.shape}",
        f"Date Range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}",
        "Columns:",
        str(list(df_clean.columns)),
        f"\nMissing Values:\n{df_clean.isnull().sum().to_string()}",
        "\nFirst 5 Rows:",
        df_clean.head().to_string()
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    print("\n" + "="*50)
    print(f"Final Dataset Shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")
    print(f"Date Range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
    print("Missing Values:")
    print(df_clean.isnull().sum())
    print(f"\nFinal feature dataset saved to {output_csv}")
    print(f"Report saved to {report_path}")
    print("="*50)

if __name__ == "__main__":
    main()
