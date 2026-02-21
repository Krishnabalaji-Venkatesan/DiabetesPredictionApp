import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.set_page_config(page_title="DIABETES PREDICTION APP", layout="wide")

background_url = "https://raw.githubusercontent.com/Krishnabalaji-Venkatesan/DiabetesPredictionApp/refs/heads/main/diabetes.jpg"

# CSS for background and Predict button
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-attachment: fixed;
    }}
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

# Centered title
st.markdown(
    '<h1 style="text-align:center; color:#1F77B4; font-weight:bold; font-size:48px;">DIABETES PREDICTION APPLICATION</h1>',
    unsafe_allow_html=True
)

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p style="font-size:24px; font-weight:bold; color:black;">GLUCOSE LEVEL</p>', unsafe_allow_html=True)
    glucose = st.number_input("", min_value=0, key="glucose", step=1, format="%d")
    
    st.markdown('<p style="font-size:24px; font-weight:bold; color:black;">BLOOD PRESSURE</p>', unsafe_allow_html=True)
    bp = st.number_input("", min_value=0, key="bp", step=1, format="%d")
    
    st.markdown('<p style="font-size:24px; font-weight:bold; color:black;">BMI</p>', unsafe_allow_html=True)
    bmi = st.number_input("", min_value=0.0, key="bmi", format="%.2f")
    
    st.markdown('<p style="font-size:24px; font-weight:bold; color:black;">AGE</p>', unsafe_allow_html=True)
    age = st.number_input("", min_value=0, key="age", step=1)

with col2:
    st.markdown('<p style="font-size:24px; font-weight:bold; color:black;">INSULIN LEVEL</p>', unsafe_allow_html=True)
    insulin = st.number_input("", min_value=0.0, key="insulin", format="%.2f")
    
    st.markdown('<p style="font-size:24px; font-weight:bold; color:black;">SKIN THICKNESS</p>', unsafe_allow_html=True)
    skin_thickness = st.number_input("", min_value=0, key="skin", step=1)
    
    st.markdown('<p style="font-size:24px; font-weight:bold; color:black;">DIABETES PEDIGREE FUNCTION</p>', unsafe_allow_html=True)
    dpf = st.number_input("", min_value=0.0, key="dpf", format="%.2f")
    
    st.markdown('<p style="font-size:24px; font-weight:bold; color:black;">PREGNANCIES</p>', unsafe_allow_html=True)
    pregnancies = st.number_input("", min_value=0, key="pregnancies", step=1)

# Centered Predict button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("PREDICT"):
        input_data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
        probability = model.predict_proba(input_data)[0][1]
        result = model.predict(input_data)[0]

        if result == 1:
            st.error(f"THE PERSON IS DIABETIC. PROBABILITY: {probability*100:.2f}%")
        else:
            st.success(f"THE PERSON IS NON-DIABETIC. PROBABILITY: {(1-probability)*100:.2f}%")

        # Bar chart
        param_names = ['PREGNANCIES','GLUCOSE','BP','SKIN THICKNESS','INSULIN','BMI','DPF','AGE']
        param_values = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
        df = pd.DataFrame({'PARAMETER': param_names, 'VALUE': param_values})

        st.subheader("HEALTH PARAMETERS OVERVIEW")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(df['PARAMETER'], df['VALUE'], color='skyblue')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
