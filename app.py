import streamlit as st
import numpy as np
import joblib
import requests
from streamlit_lottie import st_lottie

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load animation from Lottie URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_jcikwtux.json")  # You can replace with another animation

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

# ----------------- Home Page -----------------
if page == "Home":
    st.title("🩺 Welcome to the Diabetes Prediction App")
    st.write("""
        This simple web app uses a machine learning model to predict whether a person is diabetic based on medical input data.
        
        Navigate to the **Prediction** page to get started.
    """)
    st_lottie(lottie_animation, speed=1, height=300, key="home")

# ----------------- Prediction Page -----------------
elif page == "Prediction":
    st.title('🧪 Diabetes Prediction')
    st.write("Enter the following medical details:")

    # Input fields
    Pregnancies = st.number_input('Pregnancies', min_value=0)
    Glucose = st.number_input('Glucose', min_value=0)
    BloodPressure = st.number_input('Blood Pressure', min_value=0)
    SkinThickness = st.number_input('Skin Thickness', min_value=0)
    Insulin = st.number_input('Insulin', min_value=0)
    BMI = st.number_input('BMI')
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function')
    Age = st.number_input('Age', min_value=0)

    if st.button('Predict'):
        input_data = np.array([
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 0:
            st.success("✅ The person is **non-diabetic**")
        else:
            st.warning("⚠️ The person is **diabetic**")

# ----------------- About Page -----------------
elif page == "About":
    st.title("ℹ️ About This App")
    st.write("""
    This is a Streamlit app for predicting diabetes using a machine learning model trained on the Pima Indians Diabetes Dataset.

    **Developed by:** Faith Kinya  
    **Tech stack:** Python, Scikit-learn, Streamlit  
    **Deployment:** Streamlit Cloud  
    """)
