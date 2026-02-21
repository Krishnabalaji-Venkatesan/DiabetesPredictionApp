import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="DIABETES PREDICTION APP", layout="wide")

# Background image
background_url = "https://raw.githubusercontent.com/Krishnabalaji-Venkatesan/DiabetesPredictionApp/main/diabetes.jpg"

# CSS Styling
st.markdown(f"""
<style>
.stApp {{
    background-image: url("{background_url}");
    background-size: cover;
    background-attachment: fixed;
}}

/* Title */
.stTitle {{
    text-align: center;
    color: #1F77B4;
    font-weight: bold;
    font-size: 48px;
    text-transform: uppercase;
    margin-bottom: 30px;
}}

/* Inputs */
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
label {{
    color: white !important;
    font-weight: bold;
    font-size: 18px;
    text-transform: uppercase;
}}

/* Predict button styled like message box, full width */
div.stButton {{
    width: 100%;
}}
div.stButton > button:first-child {{
    display: block;
    background-color: white;
    color: #1F77B4;
    height: 100px;             /* same height as message box */
    width: 100%;               /* end-to-end */
    border-radius: 12px;
    font-size: 28px;
    font-weight: bold;
    margin-top: 25px;
    border: 3px solid #1F77B4;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.4);
}}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="stTitle">DIABETES PREDICTION APPLICATION</h1>', unsafe_allow_html=True)

# Inputs
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

# Predict button
if st.button("PREDICT"):
    input_data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
    probability = model.predict_proba(input_data)[0][1]
    result = model.predict(input_data)[0]

    # Result box with polished wording
    if result == 1:
        st.markdown(
            f"<div style='background-color:#ffcccc; padding:20px; border-radius:12px; text-align:center; font-size:22px; font-weight:bold; color:red;'>"
            f"Prediction: Diabetic (Confidence: {probability*100:.2f}%)</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color:white; padding:20px; border-radius:12px; text-align:center; font-size:22px; font-weight:bold; color:green;'>"
            f"Prediction: Non‑Diabetic (Confidence: {(1-probability)*100:.2f}%)</div>",
            unsafe_allow_html=True
        )

    # Tab-sized graph (slightly bigger, smaller font, centered)
    st.markdown("<h3 style='text-align:center; color:white;'>HEALTH PARAMETERS OVERVIEW</h3>", unsafe_allow_html=True)
    param_names = ['PREGNANCIES','GLUCOSE','BP','SKIN THICKNESS','INSULIN','BMI','DPF','AGE']
    param_values = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
    df = pd.DataFrame({'PARAMETER': param_names, 'VALUE': param_values})

    fig, ax = plt.subplots(figsize=(6,3))   # slightly bigger than before
    ax.bar(df['PARAMETER'], df['VALUE'], color='skyblue')
    ax.set_xticklabels(df['PARAMETER'], rotation=30, ha='right', fontsize=8)  # reduced font size
    ax.tick_params(axis='y', labelsize=8)  # smaller y-axis font
    plt.tight_layout()

    # Center the graph
    colA, colB, colC = st.columns([1,2,1])
    with colB:
        st.pyplot(fig)
