import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def clean_and_merge():
    print("1. Loading datasets...")
    power_df = pd.read_csv('../data/power.csv')
    env_df = pd.read_csv('../data/env.csv')

    print("2. Inspecting initial data...")
    power_df.columns = [c.strip() for c in power_df.columns]
    env_df.columns = [c.strip() for c in env_df.columns]
    env_df.rename(columns=lambda x: 'env_levelsolar' if 'env_levelsolar' in x else x, inplace=True)
    
    print(f"Power Rows: {len(power_df)}. Env Rows: {len(env_df)}")
    print(f"Power Missing:\n{power_df.isnull().sum().head()}\n")

    print("3. Combining date and time into datetime columns...")
    power_df['datetime'] = pd.to_datetime(power_df['pow_date'] + ' ' + power_df['pow_time'], errors='coerce')
    env_df['datetime'] = pd.to_datetime(env_df['env_date'] + ' ' + env_df['env_time'], errors='coerce')

    power_df = power_df.dropna(subset=['datetime'])
    env_df = env_df.dropna(subset=['datetime'])

    # Drop physically impossible dates (e.g., year 0001)
    power_df = power_df[power_df['datetime'].dt.year > 2000].copy()
    env_df = env_df[env_df['datetime'].dt.year > 2000].copy()

    print("4. Sorting by datetime...")
    power_df = power_df.sort_values('datetime').reset_index(drop=True)
    env_df = env_df.sort_values('datetime').reset_index(drop=True)

    print("5. Checking for duplicates and irregular intervals...")
    power_dupes = power_df.duplicated(subset=['datetime']).sum()
    env_dupes = env_df.duplicated(subset=['datetime']).sum()
    print(f"Power Duplicates: {power_dupes}, Env Duplicates: {env_dupes}")
    
    power_df = power_df.drop_duplicates(subset=['datetime'])
    env_df = env_df.drop_duplicates(subset=['datetime'])

    power_df.set_index('datetime', inplace=True)
    env_df.set_index('datetime', inplace=True)

    print("\n6. Datetime ranges:")
    print(f"Power: {power_df.index.min()} to {power_df.index.max()} ({len(power_df)} rows)")
    print(f"Env:   {env_df.index.min()} to {env_df.index.max()} ({len(env_df)} rows)")
    print(f"Power Most Common Interval: {power_df.index.to_series().diff().mode().iloc[0]}")
    print(f"Env Most Common Interval: {env_df.index.to_series().diff().mode().iloc[0]}")

    print("\n7 & 8. Restricting to true overlap range...")
    overlap_start = max(power_df.index.min(), env_df.index.min())
    overlap_end = min(power_df.index.max(), env_df.index.max())
    print(f"Overlap: {overlap_start} to {overlap_end}")

    power_df = power_df[(power_df.index >= overlap_start) & (power_df.index <= overlap_end)].copy()
    env_df = env_df[(env_df.index >= overlap_start) & (env_df.index <= overlap_end)].copy()

    print("\n9. Dropping fully empty or useless columns...")
    cols_to_drop = ['pow_dev_totpower', 'pow_inv_id', 'pow_date', 'pow_time', 'env_date', 'env_time']
    power_df = power_df.drop(columns=[c for c in cols_to_drop if c in power_df.columns])
    env_df = env_df.drop(columns=[c for c in cols_to_drop if c in env_df.columns])

    print("\n10 & 11. Identifying and neutralizing physically invalid values...")
    # Clean numeric columns appropriately handling strings
    for col in power_df.columns:
        if col not in ['pow_index', 'pow_id']:
            power_df[col] = pd.to_numeric(power_df[col], errors='coerce')
    for col in env_df.columns:
        if col not in ['env_index']:
            env_df[col] = pd.to_numeric(env_df[col], errors='coerce')

    # Convert known invalid bounds to NaN
    # e.g., Negative active power, impossible frequencies or temps
    if 'pow_acp' in power_df.columns:
        power_df.loc[power_df['pow_acp'] < 0, 'pow_acp'] = np.nan
    if 'pow_freq' in power_df.columns:
        power_df.loc[(power_df['pow_freq'] < 45) | (power_df['pow_freq'] > 65), 'pow_freq'] = np.nan
    
    if 'env_slopesolar' in env_df.columns:
        env_df.loc[env_df['env_slopesolar'] < 0, 'env_slopesolar'] = np.nan
    if 'env_airtemp' in env_df.columns:
        env_df.loc[(env_df['env_airtemp'] < -50) | (env_df['env_airtemp'] > 80), 'env_airtemp'] = np.nan

    print("\n12. Outliers (Extreme Outliers via standard logic)...")
    # To prevent blowing away legitimate solar peaks, we defer IQR to later, 
    # but here we could handle massive statistical spikes. We'll rely on IQR after alignment.

    print("\n13-17. Sequence Modeling Prep: Creating regular grid and merging...")
    # For recurrent modeling (LSTM), we need a regular time interval.
    # The most common power delta is ~10 min.
    # We create a perfect 10-minute grid over the overlap period.
    regular_grid = pd.date_range(start=overlap_start.ceil('10min'), 
                                 end=overlap_end.floor('10min'), 
                                 freq='10min')
    grid_df = pd.DataFrame(index=regular_grid)

    print("Merging power onto 10-minute interval with 15-minute tolerance...")
    power_df = power_df.sort_index()
    env_df = env_df.sort_index()

    # merge_asof aligns the nearest prior/exact reading within tolerance
    merged_power = pd.merge_asof(grid_df, power_df, left_index=True, right_index=True, direction='nearest', tolerance=pd.Timedelta('15min'))
    
    print("Merging env onto merged base...")
    final_df = pd.merge_asof(merged_power, env_df, left_index=True, right_index=True, direction='nearest', tolerance=pd.Timedelta('15min'))

    print("\n18. Missing values introduced by regular grid alignment:")
    print(final_df[['pow_acp', 'env_slopesolar']].isnull().sum())

    print("\n19. Interpolating small gaps only...")
    # Rule: Max gap size to interpolate is 2 hours (12 periods of 10 min)
    # This specifically addresses the 'long straight line across days' issue.
    # We use time-based interpolation up to exactly limits.
    max_gap_periods = 12 
    final_df = final_df.interpolate(method='time', limit=max_gap_periods)

    print("\nAfter interpolation, remaining continuous missing points:")
    print(final_df[['pow_acp', 'env_slopesolar']].isnull().sum())

    print("\n20. Applying Conservative IQR clipping to merged dataset...")
    # E.g. anything > Q3 + 3*IQR is clipped (3 to be very conservative for real sunlight patterns)
    def clip_iqr(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr
        lower_bound = q1 - 3 * iqr
        # Rather than dropping, we cap (clip)
        return series.clip(lower=lower_bound, upper=upper_bound)

    if 'pow_acp' in final_df.columns:
        final_df['pow_acp'] = clip_iqr(final_df['pow_acp'])
    if 'env_slopesolar' in final_df.columns:
        final_df['env_slopesolar'] = clip_iqr(final_df['env_slopesolar'])

    print("\nSaving final dataset to CSV...")
    os.makedirs('../data', exist_ok=True)
    final_df.to_csv('../data/cleaned_merged.csv')

    print("\n22 & 23. Verification Plot generation...")
    # Take a 2-week slice with data to demonstrate
    # Let's find first month with enough valid data
    valid_data = final_df.dropna(subset=['pow_acp', 'env_slopesolar'])
    if not valid_data.empty:
        start_plot = valid_data.index[0]
        end_plot = start_plot + pd.Timedelta(days=14)
        plot_df = final_df[(final_df.index >= start_plot) & (final_df.index <= end_plot)]
        
        fig, ax1 = plt.subplots(figsize=(15, 6))
        
        ax1.set_xlabel('Date Time (10-minute intervals)')
        ax1.set_ylabel('Active Power AC (pow_acp)', color='tab:blue')
        ax1.plot(plot_df.index, plot_df['pow_acp'], color='tab:blue', alpha=0.7, label='AC Power')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()  
        ax2.set_ylabel('Slope Solar Radiation (env_slopesolar)', color='tab:orange')  
        ax2.plot(plot_df.index, plot_df['env_slopesolar'], color='tab:orange', alpha=0.7, label='Solar Radiation')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        fig.tight_layout()
        plt.title('Verification of Cleaned Dataset (2-Week Realistic Solar Pattern Check)')
        # Plotting the raw NaNs that we leave mathematically blank intentionally
        plt.savefig('verification_plot.png')
        plt.close()
        print("Plot saved to analysis/verification_plot.png")
    else:
         print("Not enough overlapping valid data to plot after restrictions!")
        
    print("\nAll tasks complete.")

if __name__ == "__main__":
    clean_and_merge()
