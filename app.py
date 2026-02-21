import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="DIABETES PREDICTION APP", layout="wide")

# Background image from GitHub raw link
background_url = "https://raw.githubusercontent.com/Krishnabalaji-Venkatesan/DiabetesPredictionApp/main/diabetes.jpg"

# CSS Styling
st.markdown(f"""
<style>
.stApp {{
    background-image: url("{background_url}");
    background-size: cover;
    background-attachment: fixed;
}}

/* Title styling */
.stTitle {{
    text-align: center;
    color: #1F77B4;
    font-weight: bold;
    font-size: 48px;
    text-transform: uppercase;
    margin-bottom: 30px;
}}

/* Input box styling with glass effect */
div[data-baseweb="input"] input {{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    color: white !important;
    font-size: 20px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.4);
    padding: 6px;
    font-weight: bold;
    margin-top: 4px;
}}

/* Labels */
label {{
    color: white !important;
    font-weight: bold;
    font-size: 18px;
    text-transform: uppercase;
    margin-bottom: 2px !important;
}}

/* Center Predict button */
div.stButton {{
    display: flex;
    justify-content: center;
}}
div.stButton > button:first-child {{
    background-color: #1F77B4;
    color: white;
    height: 55px;
    width: 240px;
    border-radius: 10px;
    font-size: 22px;
    margin-top: 25px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
}}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="stTitle">DIABETES PREDICTION APPLICATION</h1>', unsafe_allow_html=True)

# Two columns for inputs
col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input("GLUCOSE LEVEL", min_value=0)
    bp = st.number_input("BLOOD PRESSURE", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    age = st.number_input("AGE", min_value=0)

with col2:
    insulin = st.number_input("INSULIN LEVEL", min_value=0.0, format="%.2f")
    skin_thickness = st.number_input("SKIN THICKNESS", min_value=0)
    dpf = st.number_input("DIABETES PEDIGREE FUNCTION", min_value=0.0, format="%.2f")
    pregnancies = st.number_input("PREGNANCIES", min_value=0)

# Predict button centered
colA, colB, colC = st.columns([1,2,1])
with colB:
    if st.button("PREDICT"):
        input_data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
        probability = model.predict_proba(input_data)[0][1]
        result = model.predict(input_data)[0]

        # Centered probability message
        if result == 1:
            st.markdown(f"<h4 style='text-align:center; color:red;'>THE PERSON IS DIABETIC. PROBABILITY: {probability*100:.2f}%</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='text-align:center; color:green;'>THE PERSON IS NON-DIABETIC. PROBABILITY: {(1-probability)*100:.2f}%</h4>", unsafe_allow_html=True)

        # Bigger bar chart, centered
        param_names = ['PREGNANCIES','GLUCOSE','BP','SKIN THICKNESS','INSULIN','BMI','DPF','AGE']
        param_values = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
        df = pd.DataFrame({'PARAMETER': param_names, 'VALUE': param_values})

        st.markdown("<h3 style='text-align:center; color:white;'>HEALTH PARAMETERS OVERVIEW</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))   # bigger graph size
        ax.bar(df['PARAMETER'], df['VALUE'], color='skyblue')
        ax.set_xticklabels(df['PARAMETER'], rotation=30, ha='right', fontsize=10)
        plt.tight_layout()
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig)


