import pandas as pd
import numpy as np

# 1. Load data
power_df = pd.read_csv('../data/power.csv')
env_df = pd.read_csv('../data/env.csv')

# Clean column names
power_df.columns = [c.strip() for c in power_df.columns]
env_df.columns = [c.strip() for c in env_df.columns]
# For env_df, fix that weird newline column name
env_df.rename(columns=lambda x: 'env_levelsolar' if 'env_levelsolar' in x else x, inplace=True)

# 2. initial inspection
print("Power Columns:", power_df.columns.tolist())
print("Env Columns:", env_df.columns.tolist())

# 3. Combine date and time
# Some dates might be invalid like 0001-01-01 in env.csv
power_df['datetime'] = pd.to_datetime(power_df['pow_date'] + ' ' + power_df['pow_time'], errors='coerce')
env_df['datetime'] = pd.to_datetime(env_df['env_date'] + ' ' + env_df['env_time'], errors='coerce')

power_df = power_df.dropna(subset=['datetime'])
env_df = env_df.dropna(subset=['datetime'])

# Filter out impossible years (like 0001)
power_df = power_df[power_df['datetime'].dt.year > 2000]
env_df = env_df[env_df['datetime'].dt.year > 2000]

# 4. Sort
power_df = power_df.sort_values('datetime').reset_index(drop=True)
env_df = env_df.sort_values('datetime').reset_index(drop=True)

# 5. Diff
power_diffs = power_df['datetime'].diff().value_counts().head()
env_diffs = env_df['datetime'].diff().value_counts().head()

print("\n--- Power Range ---")
print("Start:", power_df['datetime'].min())
print("End:", power_df['datetime'].max())
print("Rows:", len(power_df))
print("Most common intervals:\n", power_diffs)

print("\n--- Env Range ---")
print("Start:", env_df['datetime'].min())
print("End:", env_df['datetime'].max())
print("Rows:", len(env_df))
print("Most common intervals:\n", env_diffs)

# Overlap
overlap_start = max(power_df['datetime'].min(), env_df['datetime'].min())
overlap_end = min(power_df['datetime'].max(), env_df['datetime'].max())

print("\n--- Overlap ---")
if overlap_start <= overlap_end:
    print(f"Overlap Start: {overlap_start}")
    print(f"Overlap End: {overlap_end}")
else:
    print("NO OVERLAP")
