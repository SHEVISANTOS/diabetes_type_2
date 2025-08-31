# src/predict.py

import joblib
import pandas as pd
import numpy as np
import json
import os

# Define paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Load model and preprocessing objects
_model = joblib.load(os.path.join(MODEL_DIR, 'final_model.pkl'))
_scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
_le_gender = joblib.load(os.path.join(MODEL_DIR, 'le_gender.pkl'))
_le_smoke = joblib.load(os.path.join(MODEL_DIR, 'le_smoking.pkl'))

# Load imputation values
with open(os.path.join(MODEL_DIR, 'imputation_values.json'), 'r') as f:
    impute_vals = json.load(f)
MEDIAN_BMI = impute_vals['MEDIAN_BMI']
MODE_SMOKING = impute_vals['MODE_SMOKING']

def predict_diabetes(input_data):
    """
    Predict diabetes risk.
    input_data: dict with keys:
        gender, age, hypertension, heart_disease,
        smoking_history, bmi, HbA1c_level, blood_glucose_level
    """
    try:
        # Handle missing fields
        age = float(input_data.get('age', 0))
        hypertension = int(input_data.get('hypertension', 0))
        heart_disease = int(input_data.get('heart_disease', 0))
        hba1c = float(input_data.get('HbA1c_level', 5.0))
        glucose = float(input_data.get('blood_glucose_level', 100))

        bmi = input_data.get('bmi')
        if bmi is None or pd.isna(bmi) or not str(bmi).strip():
            bmi = MEDIAN_BMI
        else:
            bmi = float(bmi)

        gender = str(input_data.get('gender', 'Female'))
        smoking = str(input_data.get('smoking_history', MODE_SMOKING))

        # Validate
        if age < 0 or age > 120:
            return {"error": "Age must be between 0 and 120"}
        if bmi < 10 or bmi > 70:
            return {"error": "BMI must be between 10 and 70"}
        if hba1c < 3 or hba1c > 15:
            return {"error": "HbA1c must be between 3 and 15"}
        if glucose < 50 or glucose > 600:
            return {"error": "Blood glucose must be between 50 and 600"}

        # Encode gender and smoking
        if gender not in _le_gender.classes_:
            gender = 'Female'
        gender_encoded = _le_gender.transform([gender])[0]

        if smoking not in _le_smoke.classes_:
            smoking = MODE_SMOKING
        smoking_encoded = _le_smoke.transform([smoking])[0]

        # Create DataFrame
        input_df = pd.DataFrame([{
            'gender': gender_encoded,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_encoded,
            'bmi': bmi,
            'HbA1c_level': hba1c,
            'blood_glucose_level': glucose
        }])

        # Scale numerical features
        num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        input_df[num_cols] = _scaler.transform(input_df[num_cols])

        # Predict
        pred = _model.predict(input_df)[0]
        proba = _model.predict_proba(input_df)[0][1]

        return {
            "diabetes": int(pred),
            "probability": float(proba),
            "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.4 else "Low"
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}