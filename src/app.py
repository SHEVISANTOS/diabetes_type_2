# src/app.py

import sys
import os

# Add project root to path so 'src' is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.predict import predict_diabetes

# App title
st.set_page_config(page_title="🩺 Diabetes Risk Predictor", layout="centered")
st.title("🩺 Type 2 Diabetes Risk Prediction")
st.markdown("Enter patient information to assess diabetes risk.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 10, 100, 45)
        hypertension = st.selectbox("Hypertension (1: Yes, 0: No)", [0, 1])
        heart_disease = st.selectbox("Heart Disease (1: Yes, 0: No)", [0, 1])

    with col2:
        smoking_history = st.selectbox(
            "Smoking History",
            ["never", "current", "former", "ever", "not current"]
        )
        bmi = st.text_input("BMI (leave blank for auto-fill)", "")
        hba1c_level = st.slider("HbA1c Level", 3.0, 15.0, 5.7)
        blood_glucose_level = st.slider("Blood Glucose Level", 50, 600, 100)

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    # Handle BMI input
    try:
        bmi_val = float(bmi) if bmi.strip() else None
    except ValueError:
        st.warning("Invalid BMI value. Using median value.")
        bmi_val = None

    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi_val,
        'HbA1c_level': hba1c_level,
        'blood_glucose_level': blood_glucose_level
    }

    with st.spinner("Analyzing..."):
        result = predict_diabetes(input_data)

    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
    else:
        st.success(f"✅ Prediction: {'High Risk' if result['diabetes'] == 1 else 'Low Risk'}")
        st.metric("Diabetes Probability", f"{result['probability']:.2%}")
        st.progress(result['probability'])

        if result['probability'] > 0.7:
            st.warning("🔴 High risk. Recommend clinical evaluation.")
        elif result['probability'] > 0.4:
            st.info("🟠 Moderate risk. Monitor HbA1c and glucose.")
        else:
            st.success("🟢 Low risk. Maintain healthy lifestyle.")

# Footer
st.markdown("---")
st.caption("💡 *Model based on clinical data. Not a medical diagnosis.*")