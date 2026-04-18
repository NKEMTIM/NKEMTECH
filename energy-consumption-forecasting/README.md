# Energy Consumption Forecasting

This project focuses on energy consumption data analysis and forecasting using time series and deep learning methods. It includes data preprocessing, exploratory data analysis, feature engineering, model validation, and future forecasting.

## Project Overview
The work compares forecasting approaches such as SARIMA and LSTM for predicting energy usage patterns and visualizing future trends.

## Repository Structure

```text
energy-consumption-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
├── docs/
├── graphs/
└── results/
Folders
code/ contains the Python scripts used for preprocessing, analysis, modeling, forecasting, and validation.
docs/ contains markdown reports and summaries.
graphs/ contains forecast and validation plots.
results/ contains CSV outputs from model predictions and evaluation.
Code Files
preprocess_data.py
eda_and_features.py
model_ts.py
lstm_forecast_future.py
sarima_forecast_future.py
forecast_both_4years.py
validate_and_plot.py
run_analysis.py
data_explorer.py
check_cols.py
check_libs.py
Graph Outputs
holdout_validation_graph.png
lstm_future_forecast_graph.png
sarima_future_forecast_graph.png
future_forecast_4years_combined_graph.png
Result Files
validation_results.csv
holdout_validation_results.csv
lstm_test_predictions.csv
lstm_future_forecast.csv
sarima_future_forecast.csv
future_forecast_4years_combined.csv
Documentation
eda_report.md
preprocessing_summary.md
summary.md
Dataset Note

The raw datasets are not included in this repository.

Dataset files used in the project:

2018Use_data.xlsx
2019Use_data.xlsx
2020Use_data.xlsx
2021Use_data (1).xlsx
2022data (2).xlsx
2023Use_data (3).xlsx
2024Use_data (4).xlsx
025Use_data.xlsx
clean_merged_energy_usage.csv
energy_usage_featured.csv
Models Used
SARIMA
LSTM

How to Run
Place the dataset files in a local data/ folder.
Install required libraries from requirements.txt.
Run the preprocessing and analysis scripts in the code/ folder.
Check generated outputs in the graphs/ and results/ folders.
Author

Mmoya Patience Nkem
GitHub: NKEMTIM

## Step 4: save all files
Press `Ctrl + S` on each open file.

## Step 5: check one important thing
From your screenshot, `requirements.txt` looks like a folder, not a file.

It should be a **file**, not a folder.

So check this:
- if `requirements.txt` is a folder, delete that folder
- then create a new file named exactly `requirements.txt`

Same for `.gitignore` if it is missing.

## Step 6: final check
Your project should now have:
- `.gitignore` file
- `requirements.txt` file
- `README.md` file