import pandas as pd
import numpy as np
import io

def evaluate_env():
    with open('evaluate_results.txt', 'w', encoding='utf-8') as f:
        f.write("--- Analyzing Environmental Variables ---\n")
        
        # Load cleaned data
        df = pd.read_csv('../data/cleaned_merged.csv')
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        total_rows = len(df)
        f.write(f"Total Rows in Cleaned Dataset: {total_rows}\n\n")

        env_cols = ['env_slopesolar', 'env_levelsolar', 'env_modtemp', 'env_airtemp']
        recommendations = {}

        for col in env_cols:
            if col not in df.columns:
                f.write(f"Column {col} not found in cleaned dataset.\n\n")
                continue
                
            col_data = df[col].dropna()
            non_missing = len(col_data)
            perc_missing = (non_missing / total_rows) * 100
            
            if non_missing > 0:
                first_valid = col_data.index.min()
                last_valid = col_data.index.max()
                unique_vals = col_data.nunique()
                col_min = col_data.min()
                col_max = col_data.max()
                col_mean = col_data.mean()
                col_median = col_data.median()
                
                # Rule of thumb for "reliable" for sequence modeling
                is_reliable = (perc_missing > 20) and (unique_vals > 5)
                
                f.write(f"Column: {col}\n")
                f.write(f"1. Total non-missing: {non_missing}\n")
                f.write(f"2. Percentage non-missing: {perc_missing:.1f}%\n")
                f.write(f"3. Valid date range: {first_valid} to {last_valid}\n")
                f.write(f"4. Unique values: {unique_vals}\n")
                f.write(f"5. Stats -> Min: {col_min}, Max: {col_max}, Mean: {col_mean:.2f}, Median: {col_median}\n")
                f.write(f"6. Reliable to keep? {'YES' if is_reliable else 'NO'}\n\n")
                
                recommendations[col] = is_reliable
            else:
                f.write(f"Column: {col}\n")
                f.write("All values are missing in cleaned dataset.\n")
                f.write(f"6. Reliable to keep? NO\n\n")
                recommendations[col] = False

        f.write("--- Final Recommendation ---\n")
        kept = [c for c, is_rel in recommendations.items() if is_rel]
        dropped = [c for c, is_rel in recommendations.items() if not is_rel]
        f.write(f"KEEP: {', '.join(kept) if kept else 'None'}\n")
        f.write(f"DROP: {', '.join(dropped) if dropped else 'None'}\n")

if __name__ == "__main__":
    evaluate_env()
