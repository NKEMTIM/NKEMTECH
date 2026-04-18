# Exploratory Data Analysis & Feature Engineering Report

## 1. Saved Clean Dataset
- Dataset successfully saved to: `c:\Users\DELL\Downloads\enegy consumption\data\clean_merged_energy_usage.csv`

## 2. Dataset Overview
- **Shape**: (2400, 7)
- **Columns**: ID, Category, Year, Month, Consumption_MWh, Month_Num, Date
- **Data Types**:
```text
ID                          int64
Category                      str
Year                        int64
Month                         str
Consumption_MWh           float64
Month_Num                   int64
Date               datetime64[us]
```
- **Date Range**: 2018-01-01 to 2025-12-01
- **Target Column**: `Consumption_MWh`

## 3 & 4. Visualizations and Interpretation

### Visual 1: Line Graph of Energy Usage over Time
![Line Graph](C:/Users/DELL/.gemini/antigravity/brain/cbf9fd33-9f23-421b-8027-c66d43279b83/artifacts/line_graph_trend.png)
- **Overall Trend**: The energy usage remains relatively stable year-over-year globally without a fierce upward or downward long-term slope.
- **Variation (Seasonality)**: There is an extremely clear repeating seasonal pattern. Consumption consistently spikes in the middle of summer (July/August) likely due to cooling, and again in winter (Dec/Jan) due to heating. The spring and autumn months form the baseline valleys.
- **Forecasting Usefulness**: This distinct seasonality is a phenomenal asset for time-series forecasting. Lags and seasonal rolling windows will map this efficiently.

### Visual 2: Histogram of the Target Column `Consumption_MWh`
![Histogram](C:/Users/DELL/.gemini/antigravity/brain/cbf9fd33-9f23-421b-8027-c66d43279b83/artifacts/histogram_target.png)
- **Distribution Shape**: The distribution is highly right-skewed. The vast majority of monthly consumption records land centrally in the 80,000 to 180,000 MWh bins.
- **Variation**: The long tail extending out past 300,000 MWh represents the top-tier populous districts operating at a different scale.
- **Forecasting Usefulness**: Because it is skewed, neural networks may struggle slightly with unscaled data. The RobustScaler decided on previously remains the correct choice for this target.

### Visual 3: Boxplot of the Target Column `Consumption_MWh`
![Boxplot](C:/Users/DELL/.gemini/antigravity/brain/cbf9fd33-9f23-421b-8027-c66d43279b83/artifacts/boxplot_target.png)
- **Outliers**: We visually confirm the massive number of statistical outliers occurring past the right-side whisker (~300,000 MWh upper bound).
- **Interpretation**: These are not anomalies in the sense of 'broken sensors' or 'bad data'. These represent authentic massive districts (e.g. Gangnam-gu).
- **Forecasting Usefulness**: Confirms we should not clip these bounds, as they are real valid ceilings required for accurate district-specific inference.

## 5. Main EDA Summary
- **Trend**: Flatly stable macro long-term baseline.
- **Seasonality**: Dual-peak yearly seasonality (Summer cooling, Winter heating).
- **Skewness**: Right-skewed distribution.
- **Anomalies**: Top outliers reflect huge districts but are valid, pristine records. No impossible zero/negative values natively exist.
- **Suitability**: *Highly Suitable* for forecasting due to regular, pristine recurring periodic cycles.

##6. Check for Advanced Feature Engineering
Given the powerful seasonality identified in the Line Graph, advanced rolling and lag features are **strictly necessary** to permit standard algorithms to catch these periodic jumps.

Creating Features cautiously to avoid leakage (grouped by Category/District ensuring future boundaries don't leak into past bounds):
- **Month**: Already inherently available (`Month_Num`).
- **Day of week**: Skipped. The data is entirely Monthly totals, making daily splits mathematically impossible and irrelevant.
- **Lag_1**: Previous Month's consumption.
- **Lag_12**: The same month's consumption from exactly one year prior (crucial for summer/winter cycle).
- **Rolling_Mean_3 & Rolling_Std_3**: 3-month trailing moving averages and variation to capture localized directional momentum without leaking the prediction month.

## 7. Featured Dataset Path
- Engineered dataset successfully saved to: `c:\Users\DELL\Downloads\enegy consumption\data\energy_usage_featured.csv`

## 8. Closing Verdict
- **Key Visual Findings**: We mathematically mapped a pure dual-peak yearly seasonality tied to extreme temperature months, coupled with a heavily skewed distribution denoting huge geographical energy disparities.
- **Engineering Necessity**: Constructing `Lag_1`, `Lag_12` and robust shifted rolling statistics was strictly necessary to inject this cyclic knowledge as tabular predictor variables without forcing an algorithm to 'guess' the seasonality.
- **Next Step Execution**: For actual predictive modeling, we must drop NAs formed by our lag window logic and train models **exclusively using the `energy_usage_featured.csv` target dataset.**
