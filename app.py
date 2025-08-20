import streamlit as st
import joblib
import pandas as pd

# Load your trained model
model = joblib.load('lung_cancer_risk_model.joblib')

# Feature list
FEATURES = [
    "AGE", "GENDER", "SMOKING", "FINGER_DISCOLORATION", "MENTAL_STRESS",
    "EXPOSURE_TO_POLLUTION", "LONG_TERM_ILLNESS", "ENERGY_LEVEL",
    "IMMUNE_WEAKNESS", "BREATHING_ISSUE", "ALCOHOL_CONSUMPTION",
    "THROAT_DISCOMFORT", "OXYGEN_SATURATION", "CHEST_TIGHTNESS",
    "FAMILY_HISTORY", "SMOKING_FAMILY_HISTORY", "STRESS_IMMUNE"
]

# Streamlit UI
st.set_page_config(page_title="Lung Cancer Risk Assessment", layout="centered")

# Inject custom CSS for background color
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #ADD8E6; /* Light Blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Lung Cancer Risk Assessment")
st.write("This tool provides an estimate based on a machine learning model. Please consult a medical professional for accurate diagnosis.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking", ["Yes", "No"])
finger_discoloration = st.selectbox("Finger Discoloration", ["Yes", "No"])
mental_stress = st.selectbox("Mental Stress", ["Yes", "No"])
pollution = st.selectbox("Exposure to Pollution", ["Yes", "No"])
illness = st.selectbox("Long-term Illness", ["Yes", "No"])
energy = st.slider("Energy Level", 0, 100, 50)
immune_weakness = st.selectbox("Immune Weakness", ["Yes", "No"])
breathing_issue = st.selectbox("Breathing Issue", ["Yes", "No"])
alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
throat_discomfort = st.selectbox("Throat Discomfort", ["Yes", "No"])
oxygen = st.number_input("Oxygen Saturation (%)", min_value=80.0, max_value=100.0, value=95.0, step=0.1)
chest_tightness = st.selectbox("Chest Tightness", ["Yes", "No"])
family_history = st.selectbox("Family History", ["Yes", "No"])
smoking_family = st.selectbox("Smoking Family History", ["Yes", "No"])
stress_immune = st.selectbox("Stress-Immune Reaction", ["Yes", "No"])

# Convert categorical inputs to numeric
def yesno_to_int(val):
    return 1 if val in ["Yes", "Male"] else 0

inputs = [
    age,
    yesno_to_int(gender),
    yesno_to_int(smoking),
    yesno_to_int(finger_discoloration),
    yesno_to_int(mental_stress),
    yesno_to_int(pollution),
    yesno_to_int(illness),
    energy,
    yesno_to_int(immune_weakness),
    yesno_to_int(breathing_issue),
    yesno_to_int(alcohol),
    yesno_to_int(throat_discomfort),
    oxygen,
    yesno_to_int(chest_tightness),
    yesno_to_int(family_history),
    yesno_to_int(smoking_family),
    yesno_to_int(stress_immune),
]

df = pd.DataFrame([inputs], columns=FEATURES)

# Predict button
if st.button("Predict Risk"):
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    risk = "High Risk" if pred == 1 else "Low Risk"
    st.subheader(risk)
    st.write(f"Probability of pulmonary disease: {prob*100:.1f}%")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #333;'>
        Developed by <b>Sadia Tabassum</b> | Thanks to Kaggle for the dataset.<br><br>
        <a href="https://github.com/SadiaPikachu" target="_blank" style="color:#0000EE; text-decoration: none;">GitHub</a> |
        <a href="https://www.linkedin.com/in/sadia-tabassum-310916369" target="_blank" style="color:#0000EE; text-decoration: none;">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
