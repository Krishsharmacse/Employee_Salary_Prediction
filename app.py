import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
encoders = joblib.load("encoder.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Work Class", sorted(encoders["workclass"].classes_))
marital_status = st.sidebar.selectbox("Marital Status", sorted(encoders["marital-status"].classes_))
occupation = st.sidebar.selectbox("Occupation", sorted(encoders["occupation"].classes_))
relationship = st.sidebar.selectbox("Relationship", sorted(encoders["relationship"].classes_))
race = st.sidebar.selectbox("Race", sorted(encoders["race"].classes_))
gender = st.sidebar.selectbox("Gender", sorted(encoders["gender"].classes_))
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", sorted(encoders["native-country"].classes_))
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
education_num = st.sidebar.slider("Education Num", 1, 16, 10)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Create DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'education-num': [education_num],
    'experience': [experience]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Encode categorical columns
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    result = ">50K" if prediction[0] == 1 else "≤50K"
    st.success(f"✅ Prediction: {result}")
