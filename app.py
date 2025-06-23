import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Diabetes Prediction Web App')
st.write("Enter the following details to predict:")

# Define user input fields
Pregnancies = st.number_input('Pregnancies', min_value=0)
Glucose = st.number_input('Glucose', min_value=0)
BloodPressure = st.number_input('Blood Pressure', min_value=0)
SkinThickness = st.number_input('Skin Thickness', min_value=0)
Insulin = st.number_input('Insulin', min_value=0)
BMI = st.number_input('BMI')
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function')
Age = st.number_input('Age', min_value=0)

# Make prediction when button is clicked
if st.button('Predict'):
    input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("The person is **non-diabetic** ✅")
    else:
        st.warning("The person is **diabetic** ⚠️")
