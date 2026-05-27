This folder contains the Data & Model work for the Urban Energy Digital Twin MVP.

I prepared the hourly building electricity load dataset, performed feature engineering, trained an XGBoost regression model, and prepared the model for predict_one() integration. The model predicts hourly electricity load for building types such as Commercial, Office, Public, and Residential, and supports risk classification using q75/q90 thresholds.

Main files:
- scripts/04_feature_engineering_xgb.py: creates time, lag, and rolling features.
- scripts/05_train_xgboost.py: trains and evaluates the XGBoost model.
- scripts/06_prepare_predict_one.py: provides predict_one() and predict_one_with_risk().
- model/xgb_energy_model.pkl: saved trained model.
- data/risk_thresholds.csv: q75/q90 thresholds for normal, warning, and danger classification.
- reports/xgb_metrics_report.txt: model evaluation results.
- reports/predict_one_all_types_test_output.txt: sample prediction outputs for all building types.

How to use:
1. Install the required Python packages: pandas, numpy, scikit-learn, xgboost.
2. Keep the folder structure the same.
3. Run scripts/06_prepare_predict_one.py to test prediction.
4. Use predict_one(features) to return predicted load in kW.
5. Use predict_one_with_risk(features) to return predicted load plus risk level.