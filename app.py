import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan scaler
with open("model_knn.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Prediksi Penyakit Liver (ILPD Dataset)")

# Sidebar Input
st.sidebar.header("Input Data Pasien")

age = st.sidebar.slider("Usia", 4, 90, 45)
gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
tb = st.sidebar.slider("Total Bilirubin", 0.1, 75.0, 1.0)
db = st.sidebar.slider("Direct Bilirubin", 0.1, 20.0, 0.5)
alkphos = st.sidebar.slider("Alkaline Phosphotase", 100, 2000, 250)
sgpt = st.sidebar.slider("Alamine Aminotransferase (SGPT)", 10, 2000, 30)
sgot = st.sidebar.slider("Aspartate Aminotransferase (SGOT)", 10, 2000, 35)
tp = st.sidebar.slider("Total Proteins", 2.0, 9.0, 6.5)
alb = st.sidebar.slider("Albumin", 0.5, 6.5, 3.0)
ag_ratio = st.sidebar.slider("Albumin and Globulin Ratio", 0.1, 2.5, 1.0)

# Konversi gender ke numerik
gender_numeric = 1 if gender == "Male" else 0

# Buat DataFrame input dengan nama kolom yang sesuai
input_df = pd.DataFrame([[
    age, gender_numeric, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio
]], columns=[
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Proteins",
    "Albumin", "Albumin_and_Globulin_Ratio"
])

# Preprocessing input
input_scaled = scaler.transform(input_df)

# Prediksi
prediction = model.predict(input_scaled)[0]

# Tampilkan hasil
st.subheader("Hasil Prediksi:")
if prediction == 1:
    st.error("Pasien Diprediksi Mengidap Penyakit Liver")
else:
    st.success("Pasien Diprediksi Sehat")
