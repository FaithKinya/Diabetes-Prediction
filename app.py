import streamlit as st
import numpy as np
import joblib
import requests
from streamlit_lottie import st_lottie
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🯪",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_model_and_scaler()

@st.cache_data
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_prediction = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_5njp3vgg.json")

st.markdown('<h1 style="text-align:center; color:#4ecdc4;">Diabetes Risk Prediction</h1>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Enter Your Health Information")
    
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
    Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=100)
    BloodPressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=80)
    SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    Insulin = st.number_input('Insulin Level (µU/mL)', min_value=0, max_value=900, value=80)
    BMI = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    Age = st.number_input('Age (years)', min_value=0, max_value=120, value=30)

    if st.button('Analyze Diabetes Risk'):
        with st.spinner('Analyzing your health data...'):
            time.sleep(2)
            input_data = np.array([
                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age
            ]).reshape(1, -1)

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_scaled)[0]
            else:
                probability = [1.0 - prediction[0], prediction[0]]
                st.warning("Note: Model does not support probability prediction. Approximating results.")

            st.markdown("---")
            st.subheader("Prediction Result")

            if prediction[0] == 0:
                st.success(f"Low Risk of Diabetes (Confidence: {probability[0] * 100:.1f}%)")
            else:
                st.error(f"High Risk of Diabetes (Risk Level: {probability[1] * 100:.1f}%)")

            st.progress(probability[1])

with col2:
    if lottie_prediction:
        st_lottie(lottie_prediction, speed=1, height=300, key="prediction_animation")
