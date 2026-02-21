import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="DIABETES PREDICTION APP", layout="wide")

# Background image from GitHub
background_url = "https://raw.githubusercontent.com/Krishnabalaji-Venkatesan/DiabetesPredictionApp/refs/heads/main/diabetes.jpg"

# Apply CSS
st.markdown(
    f"""
    <style>
    /* Background image */
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-attachment: fixed;
    }}

    /* Transparent input boxes (works in Streamlit 1.30+) */
    div[data-baseweb="input"] input {{
        background: rgba(255,255,255,0.05) !important;
        color: black !important;
        font-size: 18px;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.3);
        padding: 8px;
    }}

    /* Input labels */
    label {{
        color: black !important;
        font-weight: bold;
        font-size: 18px;
        text-transform: uppercase;
    }}

    /* Remove spinner buttons background */
    input::-webkit-inner-spin-button,
    input::-webkit-outer-spin-button {{
        -webkit-appearance: none;
        margin: 0;
        background: rgba(255,255,255,0.05);
        color: black;
    }}
    input::-moz-inner-spin-button,
    input::-moz-outer-spin-button {{
        appearance: none;
        margin: 0;
        background: rgba(255,255,255,0.05);
        color: black;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Centered Title
st.markdown('<h1 style="text-align:center; color:#1F77B4; font-weight:bold; font-size:48px; text-transform:uppercase;">DIABETES PREDICTION APPLICATION</h1>', unsafe_allow_html=True)

# Input columns
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

# Center Predict button using columns
predict_col1, predict_col2, predict_col3 = st.columns([1,2,1])
with predict_col2:
    if st.button("PREDICT"):
        input_data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
        probability = model.predict_proba(input_data)[0][1]
        result = model.predict(input_data)[0]

        if result == 1:
            st.error(f"THE PERSON IS DIABETIC. PROBABILITY: {probability*100:.2f}%")
        else:
            st.success(f"THE PERSON IS NON-DIABETIC. PROBABILITY: {(1-probability)*100:.2f}%")

        # Bar chart of input parameters
        param_names = ['PREGNANCIES','GLUCOSE','BP','SKIN THICKNESS','INSULIN','BMI','DPF','AGE']
        param_values = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
        df = pd.DataFrame({'PARAMETER': param_names, 'VALUE': param_values})

        st.subheader("HEALTH PARAMETERS OVERVIEW")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(df['PARAMETER'], df['VALUE'], color='skyblue')
        plt.xticks(rotation=45)
        st.pyplot(fig)
