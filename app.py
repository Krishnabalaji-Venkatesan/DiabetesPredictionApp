import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.set_page_config(page_title="DIABETES PREDICTION APP", layout="wide")

background_url = "https://raw.githubusercontent.com/Krishnabalaji-Venkatesan/DiabetesPredictionApp/refs/heads/main/diabetes.jpg"

# CSS for background, inputs, and spacing
st.markdown(f"""
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
/* Black input boxes with white numbers */
div[data-baseweb="input"] input {{
    background: #000000 !important;  
    color: white !important;
    font-size: 24px;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.7);
    padding: 8px;
    font-weight: bold;
    margin-top: 0px !important;
    margin-bottom: 5px !important;
}}
</style>
""", unsafe_allow_html=True)

# Centered title
st.markdown('<h1 style="text-align:center; color:#1F77B4; font-weight:bold; font-size:48px;">DIABETES PREDICTION APPLICATION</h1>', unsafe_allow_html=True)

# Function to create label + input together
def number_input_box(label, **kwargs):
    st.markdown(f'<p style="font-size:24px; font-weight:bold; color:black; margin-bottom:2px;">{label}</p>', unsafe_allow_html=True)
    return st.number_input("", **kwargs)

# Two columns
col1, col2 = st.columns(2)

with col1:
    glucose = number_input_box("GLUCOSE LEVEL", min_value=0, key="glucose")
    bp = number_input_box("BLOOD PRESSURE", min_value=0, key="bp")
    bmi = number_input_box("BMI", min_value=0.0, key="bmi", format="%.2f")
    age = number_input_box("AGE", min_value=0, key="age")

with col2:
    insulin = number_input_box("INSULIN LEVEL", min_value=0.0, key="insulin", format="%.2f")
    skin_thickness = number_input_box("SKIN THICKNESS", min_value=0, key="skin")
    dpf = number_input_box("DIABETES PEDIGREE FUNCTION", min_value=0.0, key="dpf", format="%.2f")
    pregnancies = number_input_box("PREGNANCIES", min_value=0, key="pregnancies")

# Centered predict button
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
