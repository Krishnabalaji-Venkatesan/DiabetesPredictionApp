import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# --- CSS Styling and Background ---
st.markdown(
    """
    <style>
    /* Background image */
    .stApp {
        background-image: url("https://www.freepik.com/free-photo/top-view-glucose-measuring-device-diabetes_65609449.htm#fromView=search&page=1&position=37&uuid=a2a5ad31-db08-4433-96a0-b8f2bc1a840d&query=Diabetes+prediction+background.jpg");
        background-size: cover;
        background-attachment: fixed;
        opacity: 0.95;
    }

    /* Title style */
    .css-10trblm {
        color: #1F77B4;
        font-size: 40px;
        font-weight: bold;
    }

    /* Button style */
    div.stButton > button:first-child {
        background-color: #1F77B4;
        color: white;
        height: 50px;
        width: 200px;
        border-radius: 10px;
        font-size: 20px;
    }

    /* Input boxes style */
    .stNumberInput>div>input {
        background-color: rgba(255,255,255,0.8);
        color: black;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("Diabetes Prediction Application")

# --- Input Fields in Two Columns ---
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

# --- Predict Button in Center ---
predict_col1, predict_col2, predict_col3 = st.columns([1,2,1])
with predict_col2:
    if st.button("Predict"):
        input_data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
        probability = model.predict_proba(input_data)[0][1]  # Probability of being diabetic
        result = model.predict(input_data)[0]

        # Display Result
        if result == 1:
            st.error(f"The person is Diabetic. Probability: {probability*100:.2f}%")
        else:
            st.success(f"The person is Non-Diabetic. Probability: {(1-probability)*100:.2f}%")

        # --- Bar Chart of Parameters ---
        param_names = ['Pregnancies','Glucose','BP','SkinThickness','Insulin','BMI','DPF','Age']
        param_values = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
        df = pd.DataFrame({'Parameter': param_names, 'Value': param_values})

        st.subheader("Health Parameters Overview")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(df['Parameter'], df['Value'], color='skyblue')
        plt.xticks(rotation=45)
        st.pyplot(fig)


