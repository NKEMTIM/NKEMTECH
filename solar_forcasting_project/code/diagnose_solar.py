import pandas as pd
import numpy as np
import io

def diagnose_solar():
    with open('diagnose_results.txt', 'w', encoding='utf-8') as f:
        f.write("--- Loading Datasets ---\n")
        cleaned_df = pd.read_csv('../data/cleaned_merged.csv')
        if 'Unnamed: 0' in cleaned_df.columns:
            cleaned_df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
        cleaned_df['datetime'] = pd.to_datetime(cleaned_df['datetime'])
        cleaned_df.set_index('datetime', inplace=True)

        total_valid = cleaned_df['env_slopesolar'].notna().sum()
        f.write(f"\n1. Total non-missing env_slopesolar in cleaned dataset: {total_valid}\n")

        start_date = pd.to_datetime('2017-05-26 12:10:00')
        end_date = start_date + pd.Timedelta(days=14)
        window_df = cleaned_df[(cleaned_df.index >= start_date) & (cleaned_df.index <= end_date)]
        window_valid = window_df['env_slopesolar'].notna().sum()
        f.write(f"2. Non-missing env_slopesolar in 14-day plot window: {window_valid}\n")

        f.write("\n3. Statistics for env_slopesolar (window over 14 days):\n")
        if window_valid > 0:
            f.write(f"   Min: {window_df['env_slopesolar'].min()}\n")
            f.write(f"   Max: {window_df['env_slopesolar'].max()}\n")
            f.write(f"   Mean: {window_df['env_slopesolar'].mean():.2f}\n")
            f.write(f"   Median: {window_df['env_slopesolar'].median()}\n")
            f.write(f"   Unique Values: {window_df['env_slopesolar'].nunique()}\n")

        f.write("\n4. First 20 timestamps where env_slopesolar > 0 or not missing:\n")
        valid_mask = cleaned_df['env_slopesolar'].notna()
        f.write(cleaned_df[valid_mask].head(20)['env_slopesolar'].to_string() + "\n")

        # Diagnose the loss: Pre-merge vs Merge Mismatch
        f.write("\n--- Diagnosing the Loss ---\n")
        raw_env = pd.read_csv('../data/env.csv')
        raw_env.columns = [c.strip() for c in raw_env.columns]
        raw_env.rename(columns=lambda x: 'env_levelsolar' if 'env_levelsolar' in x else x, inplace=True)
        
        raw_env['datetime'] = pd.to_datetime(raw_env['env_date'] + ' ' + raw_env['env_time'], errors='coerce')
        raw_env = raw_env.dropna(subset=['datetime'])
        raw_env.set_index('datetime', inplace=True)
        raw_env = raw_env.sort_index()

        raw_window = raw_env[(raw_env.index >= start_date) & (raw_env.index <= end_date)].copy()
        raw_window['env_slopesolar'] = pd.to_numeric(raw_window['env_slopesolar'], errors='coerce')
        
        raw_invalid = (raw_window['env_slopesolar'] < 0).sum()
        raw_valid_before_clip = raw_window['env_slopesolar'].notna().sum() - raw_invalid

        f.write(f"Raw env.csv has {len(raw_window)} rows in the 14-day window.\n")
        f.write(f"Raw valid env_slopesolar (>=0 and not NA) in window: {raw_valid_before_clip}\n")
        
        if raw_valid_before_clip < 100:
            conclusion = "The data was already mostly missing or invalid in the original env.csv file for this specific 2-week period. The merge did not break the data; the sensor was likely offline."
        else:
            conclusion = "The raw env.csv file contains plenty of data. The loss happened during the `merge_asof` step, likely because the 15-minute tolerance couldn't match timestamps, or timestamps were severely disconnected between power and env datasets."

        f.write("\n7. Conclusion:\n")
        f.write(conclusion + "\n")

if __name__ == "__main__":
    diagnose_solar()
