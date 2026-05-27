import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

# Paths
BASE_DIR = Path(r"c:\Users\Administrator\Downloads\Energy Forecasting\analysis")
INPUTS_DIR = BASE_DIR / "model_inputs"
OUTPUTS_DIR = BASE_DIR / "model_outputs"

def predict_one(features: dict) -> float:
    """
    - load xgb_energy_model.pkl
    - accept one input row containing the same model features used during training
    - return: predicted_kw (float)
    """
    with open(OUTPUTS_DIR / "xgb_energy_model.pkl", "rb") as f:
        model_dict = pickle.load(f)
        
    model = model_dict['model']
    le = model_dict['label_encoder']
    features_list = model_dict['features']
    
    # Validation: missing required features
    missing_features = [f for f in features_list if f not in features]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
        
    b_type = features.get('building_type')
    
    # Validation: unknown building_type
    if b_type not in le.classes_:
        raise ValueError(f"Unknown building_type: {b_type}")
        
    # Prepare input DataFrame for the model
    df_in = pd.DataFrame([features])
    
    # Label encode building_type as it was done in training
    df_in['building_type'] = le.transform([b_type])[0]
        
    # Ensure only the features used during training are passed to the model (and in exactly the same order)
    X_in = df_in[features_list]
    
    pred_kw = float(model.predict(X_in)[0])
    
    return round(pred_kw, 2)

def predict_one_with_risk(features: dict) -> dict:
    """
    - calls predict_one(features)
    - loads risk_thresholds.csv
    - returns dict with building_type, predicted_kw, and risk_level
    """
    pred_kw = predict_one(features)
    
    b_type = features['building_type']
    
    risk_df = pd.read_csv(OUTPUTS_DIR / "risk_thresholds.csv")
    b_risk = risk_df[risk_df['building_type'] == b_type]
    
    if b_risk.empty:
        raise ValueError(f"No risk thresholds found for building_type: {b_type}")
        
    b_risk = b_risk.iloc[0]
    q75 = b_risk['q75']
    q90 = b_risk['q90']
    
    if pred_kw < q75:
        risk_level = 'normal'
    elif pred_kw < q90:
        risk_level = 'warning'
    else:
        risk_level = 'danger'
        
    return {
        'building_type': b_type,
        'predicted_kw': pred_kw,
        'risk_level': risk_level
    }

def main():
    print("Testing functions...")
    df = pd.read_csv(INPUTS_DIR / "energy_features_xgb.csv")
    
    # 6. Test both functions using one sample row
    sample_row = df.iloc[0].to_dict()
    
    print("Testing predict_one()...")
    pred_val = predict_one(sample_row)
    print(f"predict_one output: {pred_val}")
    
    print("Testing predict_one_with_risk()...")
    pred_risk = predict_one_with_risk(sample_row)
    print(f"predict_one_with_risk output: {pred_risk}")
    
    print("Testing all building types...")
    # 7. Test one sample from each building type
    building_types = ['Commercial', 'Office', 'Public', 'Residential']
    test_outputs = []
    
    for b_type in building_types:
        sample = df[df['building_type'] == b_type].iloc[0].to_dict()
        try:
            res = predict_one_with_risk(sample)
            test_outputs.append({
                "building_type": b_type,
                "input_sample": sample,
                "output": res
            })
            print(f"Tested {b_type} successfully.")
        except Exception as e:
            print(f"Error testing {b_type}: {e}")
            
    print("Saving test output...")
    # 9. Save multi-building test output
    with open(OUTPUTS_DIR / "predict_one_all_types_test_output.txt", 'w', encoding='utf-8') as f:
        for res in test_outputs:
            f.write(f"Building Type: {res['building_type']}\n")
            f.write("Prediction Output:\n")
            f.write(json.dumps(res['output'], indent=2, default=str) + "\n")
            f.write("-" * 40 + "\n")
            
    print("Saving report...")
    # 10. Save updated readiness report
    report_content = (
        "Step 6: predict_one Ready Report\n"
        "================================\n\n"
        "- predict_one(features: dict) returns only predicted_kw as a float\n"
        "- predict_one_with_risk(features: dict) returns predicted_kw plus risk_level\n"
        "- the interface is ready for FastAPI backend integration\n"
        "- backend must provide the same feature columns used during model training\n"
    )
    
    with open(OUTPUTS_DIR / "step6_predict_one_ready_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print("Done! predict_one is ready.")

if __name__ == "__main__":
    main()
