import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Prediction App")

# User input fields
preg = st.number_input("Pregnancies", 0, 20)
glu = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 140)
skin = st.number_input("Skin Thickness", 0, 100)
ins = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 120)

input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
prediction = model.predict(input_data)

if st.button("Predict"):
    if prediction[0]==1:
        st.success("The person is Diabetic")
    else:
        st.success("The person is Not Diabetic")