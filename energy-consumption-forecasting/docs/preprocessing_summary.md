# Energy Data Preprocessing Report

## 1 & 2. Loading and Merged Data Check

- **Rows merged correctly?** Yes, row count equals sum of all files (26 * 8 = 208).
- **Shared columns aligned?** Yes.
- **Dropped/Unmatched/Renamed Columns?** None. All files had the exact same structure.
- **Duplicate rows created during merging?** 0

## 3. Data Cleaning

- **Removed aggregate row**: Dropped the grand total row (`총합`) from each year, reducing rows from 208 to 200.
- **Unpivoting (Melting)**: Converted 12 month columns into proper strict rows. Result shape: (2400, 7).
- **Created 'Date'**: Engineered start-of-month valid datetimes bridging 'Year' and 'Month'.
- **Final Duplicate Exact Rows**: 0

## 4. Handle Missing Values

Missing Values Count before imputation:
```text
                 Count  Percentage (%)
ID                   0             0.0
Category             0             0.0
Year                 0             0.0
Month                0             0.0
Consumption_MWh      0             0.0
Month_Num            0             0.0
Date                 0             0.0
```

- **Missingness Explanation**: As seen above, there is 0% missing data natively. No forward fill, backward fill, or imputation algorithms are required. We leave all columns as is.

## 5. Check the Target Energy-Usage Column

- **Target Column**: `Consumption_MWh`
- **Missing**: 0
- **Suspicious Zeros/Negatives**: 0
- **Outliers Count (1.5*IQR)**: 122
- **Cleaning Decision**: Since these outliers represent true, massive usage disparities between small districts and major city centers (like Gangnam), they are genuine signals and should **NOT** be capped or scaled out artificially. We leave them unchanged.

## 6. Feature Scaling

Numeric features to base our decision on: `Consumption_MWh` (Target Univariate Feature).
- Do not scale datetimes (`Date`).
- Do not scale IDs / Categoricals (`ID`, `Year`, `Month_Num`, `Category`).

### Exploring Scaling Choices:
1. **StandardScaler**: Assumes normal distribution. Because our district usages vary wildly and are right-skewed, this may distort smaller scale variances.
2. **MinMaxScaler**: Scales all data between 0 and 1. Highly effective for deep learning models (LSTMs) but can compress variance if massive un-capped outliers exist.
3. **RobustScaler**: Uses median and interquartile range (IQR). Excels when outliers are present, preventing them from shifting the mean heavily.

### Chosen Approach:
**RobustScaler** is the safest best approach here globally since we officially have massive true outliers (huge districts vs small districts). RobustScaler ensures that the bulk of normal district usages aren't squeezed into microscopic fractions due to massive values from top districts like Gangnam-gu.

Scaled Target DataFrame head:
```text
  Category       Date  Consumption_MWh  Consumption_MWh_Scaled
0      강남구 2018-01-01       415373.478                4.001232
1      강남구 2018-02-01       414856.181                3.994073
2      강남구 2018-03-01       332409.257                2.853093
3      강남구 2018-04-01       306783.400                2.498458
4      강남구 2018-05-01       294257.090                2.325107
```

### Final List of Features:
- **Unscaled**: `ID`, `Category`, `Year`, `Month`, `Month_Num`, `Date`, `Consumption_MWh`
- **Scaled**: `Consumption_MWh_Scaled`

## 7. Preprocessing Summary

- **What was cleaned**: Removed top-level aggregate ('총합') preventing duplication of totals. Unpivoted the table for strict Time-Series flow.
- **Handling Missing Values**: Data was 100% complete natively, zero imputation was necessary.
- **Merging Verification**: Confirmed standard lengths and identical structure across all 8 chronological files.
- **Scaler Chosen**: `RobustScaler` applied to target `Consumption_MWh`, minimizing scaling disruption caused by enormous differences in district populations/usages.
- **Next Stage Status**: The dataset is fully cleaned, chronologically bound, transformed into time-series form, and formally ready for statistical modeling or split validation.
