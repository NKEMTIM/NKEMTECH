import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_verification():
    # 1. Use the cleaned merged dataframe
    df = pd.read_csv('../data/cleaned_merged.csv')

    # 12. Make sure the datetime column is correctly parsed and sorted
    # The datetime column from our previous script might be 'datetime' or an unnamed index column.
    # Looking at the previous script, it saved the index, so let's check columns.
    if 'datetime' not in df.columns and 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 11. Drop rows where both pow_acp and env_slopesolar are missing
    df = df.dropna(subset=['pow_acp', 'env_slopesolar'], how='all')

    # 5. Show only a limited time window (e.g., 14 days)
    if not df.empty:
        start_date = df['datetime'].min()
        end_date = start_date + pd.Timedelta(days=14)
        plot_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        
        # 10. Make the figure large enough
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # 6. Add a clear title
        plt.title('14-Day Dual-Axis Verification Plot: Solar Radiation vs AC Power', fontsize=16, pad=15)

        # 3. Plot pow_acp on left y-axis as blue line
        # 2. Use datetime on x-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Datetime', fontsize=12)
        ax1.set_ylabel('AC Power (pow_acp)', color=color1, fontsize=12)
        ax1.plot(plot_df['datetime'], plot_df['pow_acp'], color=color1, label='AC Power', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # 9. Add grid lines
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 4. Plot env_slopesolar on right y-axis as orange line
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Solar Radiation (env_slopesolar)', color=color2, fontsize=12)
        ax2.plot(plot_df['datetime'], plot_df['env_slopesolar'], color=color2, label='Solar Radiation', linewidth=2, alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color2)

        # 8. Rotate x-axis dates so they are readable
        # Format the dates on x-axis nicely
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        
        # 13. Save the graph
        plt.savefig('verification_plot.png', dpi=150)
        plt.close()

        print("\n--- Plot Statistics ---")
        print(f"Date Range used: {plot_df['datetime'].min()} to {plot_df['datetime'].max()}")
        print(f"Non-missing pow_acp values in window: {plot_df['pow_acp'].notna().sum()}")
        print(f"Non-missing env_slopesolar values in window: {plot_df['env_slopesolar'].notna().sum()}")
        print("Graph successfully saved as analysis/verification_plot.png")
    else:
        print("Dataframe is empty after dropping missing rows. Cannot plot.")

if __name__ == "__main__":
    plot_verification()
