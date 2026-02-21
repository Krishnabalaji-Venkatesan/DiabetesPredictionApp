import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

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

    /* Title styling: bold + black */
    .stTitle {{
        text-align: center;
        color: black;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 20px;
    }}

    /* Transparent input boxes with black text */
    .stNumberInput>div>div>input {{
        background-color: rgba(255, 255, 255, 0.5);
        color: black;
        font-size: 16px;
    }}

    /* Input labels in black */
    label {{
        color: black;
        font-weight: bold;
    }}

    /* Center Predict button */
    div.stButton > button:first-child {{
        background-color: #1F77B4;
        color: white;
        height: 50px;
        width: 220px;
        border-radius: 10px;
        font-size: 20px;
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-top: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Centered Title
st.markdown('<h1 class="stTitle">Diabetes Prediction Application</h1>', unsafe_allow_html=True)

# Two input columns
col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input("Glucose Level", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    age = st.number_input("Age", min_value=0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0.0, format="%.2f")
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.2f")
    pregnancies = st.number_input("Pregnancies", min_value=0)

# Predict button
if st.button("Predict"):
    input_data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
    probability = model.predict_proba(input_data)[0][1]
    result = model.predict(input_data)[0]

    if result == 1:
        st.error(f"The person is Diabetic. Probability: {probability*100:.2f}%")
    else:
        st.success(f"The person is Non-Diabetic. Probability: {(1-probability)*100:.2f}%")

    # Bar chart of input parameters
    param_names = ['Pregnancies','Glucose','BP','SkinThickness','Insulin','BMI','DPF','Age']
    param_values = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
    df = pd.DataFrame({'Parameter': param_names, 'Value': param_values})

    st.subheader("Health Parameters Overview")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(df['Parameter'], df['Value'], color='skyblue')
    plt.xticks(rotation=45)
    st.pyplot(fig)
