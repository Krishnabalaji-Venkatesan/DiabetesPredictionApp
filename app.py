import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="DIABETES PREDICTION APP", layout="wide")

# Background image
background_url = "https://raw.githubusercontent.com/Krishnabalaji-Venkatesan/DiabetesPredictionApp/refs/heads/main/diabetes.jpg"

# CSS Styling
st.markdown(
    f"""
    <style>
    /* Background image */
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-attachment: fixed;
    }}

    /* Title styling: bold + blue + uppercase */
    .stTitle {{
        text-align: center;
        color: #1F77B4;   /* Same blue as Predict button */
        font-size: 48px;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 30px;
    }}

    /* Glassmorphism input boxes */
    input[type="number"] {{
        background: rgba(255, 255, 255, 0.05); /* super transparent */
        backdrop-filter: blur(6px);             /* blur background behind box */
        -webkit-backdrop-filter: blur(6px);
        color: black;
        font-size: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 2px 2px 10px rgba(0,0,0,0.25);
        padding: 8px;
    }}

    /* Remove dark shading from +/- spinner controls */
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button {{
        -webkit-appearance: none;
        background: transparent;
        margin: 0;
    }}
    input[type=number]::-moz-inner-spin-button,
    input[type=number]::-moz-outer-spin-button {{
        appearance: none;
        background: transparent;
        margin: 0;
    }}

    /* Input labels in uppercase + black + bigger size */
    label {{
        color: black !important;
        font-weight: bold;
        font-size: 18px;
        text-transform: uppercase;
    }}

    /* Center Predict button with flexbox */
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
    """,
    unsafe_allow_html=True
)

# Centered Title
st.markdown('<h1 class="stTitle">DIABETES PREDICTION APPLICATION</h1>', unsafe_allow_html=True)

# Two input columns
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
