import streamlit as st
import numpy as np
import joblib
import requests
from streamlit_lottie import st_lottie

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# Load Lottie animation from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_home = load_lottieurl("https://lottie.host/edc1de1f-56da-4865-8d59-2fc6cf8ef3b0/pJdw7O9BrF.json")

# Sidebar navigation
st.sidebar.title("📍 Navigate")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧪 Prediction", "ℹ️ About"])

# ---------------- Home Page ----------------
if page.startswith("🏠"):
    st.title("👋 Welcome to the Diabetes Risk Checker!")
    st.subheader("A friendly and simple ML-powered tool to assess diabetes risk.")
    st.write("""
    Diabetes is a common condition, but early detection helps with prevention and management.
    
    This app uses a trained machine learning model to help you check for possible risk — **based on medical inputs** like glucose, BMI, and more.
    
    👉 Go to the **Prediction** page to get started.
    """)
    st_lottie(lottie_home, height=300, key="home_anim")

# ---------------- Prediction Page ----------------
elif page.startswith("🧪"):
    st.title("🧪 Diabetes Risk Prediction")

    st.markdown("Please fill in your medical details below:")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input('👶 Number of Pregnancies', min_value=0)
        Glucose = st.number_input('🍬 Glucose Level', min_value=0)
        BloodPressure = st.number_input('💓 Blood Pressure', min_value=0)
        SkinThickness = st.number_input('📏 Skin Thickness', min_value=0)

    with col2:
        Insulin = st.number_input('💉 Insulin Level', min_value=0)
        BMI = st.number_input('⚖️ Body Mass Index (BMI)')
        DiabetesPedigreeFunction = st.number_input('🧬 Diabetes Pedigree Function')
        Age = st.number_input('🎂 Age', min_value=0)

    if st.button("🔍 Predict"):
        input_data = np.array([
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.markdown("## 🔎 Result:")
        if prediction[0] == 0:
            st.success("🎉 You are likely **non-diabetic**. Keep maintaining a healthy lifestyle!")
        else:
            st.error("⚠️ You may be at **risk for diabetes**. Please consult a medical professional.")

# ---------------- About Page ----------------
elif page.startswith("ℹ️"):
    st.title("ℹ️ About This App")
    st.markdown("""
    This app was created as a mini project using:
    
    - ✅ **Python & Machine Learning**
    - ✅ **Streamlit for the user interface**
    - ✅ **Pima Indians Diabetes Dataset** for training the model
    
    **Why this app?**  
    Because early awareness can save lives. This tool is meant to empower everyday people to take a small step toward their health journey.

    ---
    **Developer:** Faith Kinya  
    **Age:** 20  
    **Mission:** Inspire young people (especially in Africa 🌍) to build, heal, and thrive using tech.
    
    ✨ Thank you for trying it out!
    """)
