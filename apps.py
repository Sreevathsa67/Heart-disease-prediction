
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import streamlit as st
import joblib


model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
st.title("Heart Disease Prediction App")



age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=120)
chol = st.number_input('Cholesterol Level (mg/dl)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)', [0, 1])
restecg = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', min_value=50, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina (0 = No, 1 = Yes)', [0, 1])
oldpeak = st.slider('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, value=1.0, step=0.1)
slope = st.selectbox('Slope of Peak Exercise ST (0-2)', [0, 1, 2])
ca = st.slider('Number of Major Vessels (0-3)', min_value=0, max_value=3, value=1)
thal = st.selectbox('Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', [1, 2, 3])

test_sample = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
X_test_sample = scaler.transform(test_sample)
if st.button('predict heart disease'):
    prediction = model.predict(X_test_sample)[0] 
    if prediction == 1:
        st.write(f"HEART Disease deteced!!")
    else:
        st.write("healthy")
