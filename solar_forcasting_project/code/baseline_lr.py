import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import io

def run_baseline():
    with open('baseline_results.txt', 'w', encoding='utf-8') as f:
        # 1. Load dataset
        df = pd.read_csv('../data/cleaned_merged.csv')
        
        # 2. Choose irradiance column
        # Both have the same coverage, but slopesolar is the radiation hitting the tilted panel,
        # perfectly matching AC power generation physically.
        feature_col = 'env_slopesolar'
        target_col = 'pow_acp'
        
        # 4. Drop rows where either is missing
        df_clean = df.dropna(subset=[feature_col, target_col])
        
        # 5. Print remaining rows
        total_rows = len(df_clean)
        f.write(f"Remaining valid rows after dropping missing data: {total_rows}\n\n")
        
        if total_rows < 100:
            f.write("WARNING: Too few valid rows to run a meaningful model. Stopping.\n")
            return
            
        # 6. Summary statistics
        f.write("--- Summary Statistics ---\n")
        f.write(f"{feature_col} (Input):\n")
        f.write(f"  Mean: {df_clean[feature_col].mean():.2f}\n")
        f.write(f"  Median: {df_clean[feature_col].median():.2f}\n")
        f.write(f"  Std Dev: {df_clean[feature_col].std():.2f}\n")
        
        f.write(f"\n{target_col} (Target):\n")
        f.write(f"  Mean: {df_clean[target_col].mean():.2f}\n")
        f.write(f"  Median: {df_clean[target_col].median():.2f}\n")
        f.write(f"  Std Dev: {df_clean[target_col].std():.2f}\n\n")
        
        # 7. Split data
        X = df_clean[[feature_col]].values
        y = df_clean[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 8. Train simple linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 9. Evaluate model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        f.write("--- Model Evaluation (Test Set) ---\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n\n")
        
        # 10. Print equation
        intercept = model.intercept_
        coef = model.coef_[0]
        f.write("--- Regression Equation ---\n")
        f.write(f"AC Power = {intercept:.2f} + {coef:.2f} × {feature_col}\n\n")
        
        # 13. Interpretation
        f.write("--- Interpretation ---\n")
        if r2 > 0.8:
            interp = f"Solar irradiance ({feature_col}) alone explains AC power exceptionally well (R² = {r2:.2f}). There is a strong, highly linear baseline physical relationship."
        elif r2 > 0.5:
            interp = f"Solar irradiance explains AC power decently well (R² = {r2:.2f}), but there is significant variance. Modtemp or other factors likely impact inverter efficiency."
        else:
            interp = f"Solar irradiance explains AC power poorly (R² = {r2:.2f}). A simple linear model is insufficient, or the relationships are heavily non-linear and impacted by other factors."
        f.write(interp + "\n")
        
        # 11. Make scatter plot
        # To avoid massive plots with 50K points overlapping, we can use alpha
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='tab:blue', alpha=0.1, s=10, label='Actual Data')
        
        # Overlay regression line
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        plt.plot(x_line, y_line, color='red', linewidth=3, label=f'Linear Fit (R²={r2:.2f})')
        
        # 12. Add labels and title
        plt.title('Baseline Linear Regression: Solar Irradiance vs AC Power', fontsize=14, pad=15)
        plt.xlabel(f'Solar Irradiance ({feature_col})', fontsize=12)
        plt.ylabel(f'AC Power ({target_col})', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('baseline_plot.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    run_baseline()
