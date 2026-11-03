from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.config import MODEL_FILE, PREPROCESSOR_FILE, FEATURE_NAMES_FILE

st.set_page_config(
    page_title="Explainable AI Disease Risk",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Explainable AI Disease Risk Prediction")
st.caption("Educational/research prototype — not a medical diagnostic tool.")

if not MODEL_FILE.exists() or not PREPROCESSOR_FILE.exists():
    st.error("Model files are missing. Run the training steps first.")
    st.code("python data/generate_data.py\npython src/train_model.py")
    st.stop()

model = joblib.load(MODEL_FILE)
preprocessor = joblib.load(PREPROCESSOR_FILE)
feature_names = joblib.load(FEATURE_NAMES_FILE)

st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 80, 40)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
bmi = st.sidebar.number_input("BMI", 15.0, 50.0, 25.0, 0.1)
systolic_bp = st.sidebar.number_input("Systolic Blood Pressure", 80, 220, 120)
glucose = st.sidebar.number_input("Glucose", 50, 300, 100)
cholesterol = st.sidebar.number_input("Cholesterol", 80, 400, 190)
physical_activity = st.sidebar.slider("Physical Activity (hours/week)", 0.0, 15.0, 3.0, 0.5)
sleep_hours = st.sidebar.slider("Sleep (hours/night)", 3.0, 12.0, 7.0, 0.5)
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
family_history = st.sidebar.selectbox("Family History", ["No", "Yes"])

input_df = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "systolic_bp": systolic_bp,
    "glucose": glucose,
    "cholesterol": cholesterol,
    "physical_activity_hours": physical_activity,
    "sleep_hours": sleep_hours,
    "smoking": smoking,
    "family_history": family_history,
}])

if st.button("Predict Disease Risk", type="primary"):
    transformed = preprocessor.transform(input_df)
    probability = float(model.predict_proba(transformed)[0, 1])
    prediction = int(probability >= 0.5)

    if probability < 0.33:
        level = "LOW"
    elif probability < 0.66:
        level = "MEDIUM"
    else:
        level = "HIGH"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Risk", level)
    with col2:
        st.metric("Risk Probability", f"{probability * 100:.1f}%")

    st.subheader("Why did the model make this prediction?")

    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)
    else:
        explainer = shap.LinearExplainer(model, transformed)
        shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
    elif getattr(shap_values, "ndim", 1) == 3:
        contributions = shap_values[0, :, 1]
    else:
        contributions = shap_values[0]

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP contribution": contributions,
    })
    explanation["Absolute contribution"] = explanation["SHAP contribution"].abs()
    explanation = explanation.sort_values(
        "Absolute contribution", ascending=False
    ).head(10)

    st.dataframe(
        explanation[["Feature", "SHAP contribution"]].reset_index(drop=True),
        use_container_width=True,
    )

    st.bar_chart(
        explanation.set_index("Feature")["SHAP contribution"]
    )

    st.info(
        "A positive SHAP value pushes the prediction toward higher risk; "
        "a negative value pushes it toward lower risk. SHAP values explain "
        "the model's behavior and do not prove medical causation."
    )

st.divider()

st.subheader("About this project")
st.write(
    "This project demonstrates machine-learning disease-risk prediction "
    "with explainable AI. It uses a synthetic dataset for demonstration. "
    "For real-world medical research, a validated and appropriately licensed "
    "clinical dataset, external validation, calibration, fairness analysis, "
    "and expert oversight are required."
)
