import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Judul
st.title("Prediksi Penyakit Liver - ILPD Dataset")

# Load model dan scaler
with open("model_knn.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Input data dari user
st.sidebar.header("Masukkan Data Pasien")

age = st.sidebar.slider("Usia", 4, 90, 45)
gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
tb = st.sidebar.slider("Total Bilirubin", 0.1, 75.0, 1.0)
db = st.sidebar.slider("Direct Bilirubin", 0.1, 20.0, 0.5)
alkphos = st.sidebar.slider("Alkaline Phosphotase", 100, 2000, 250)
sgpt = st.sidebar.slider("SGPT", 10, 2000, 30)
sgot = st.sidebar.slider("SGOT", 10, 2000, 35)
tp = st.sidebar.slider("Total Protein", 2.0, 9.0, 6.5)
alb = st.sidebar.slider("Albumin", 0.5, 6.5, 3.0)
ag_ratio = st.sidebar.slider("A/G Ratio", 0.1, 2.5, 1.0)

# Buat dataframe dari input
input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "TB": [tb],
    "DB": [db],
    "Alkphos": [alkphos],
    "SGPT": [sgpt],
    "SGOT": [sgot],
    "TP": [tp],
    "ALB": [alb],
    "A/G Ratio": [ag_ratio]
})

# Mapping gender
input_df["Gender"] = input_df["Gender"].map({"Male": 1, "Female": 0})

# Pastikan urutan kolom sama seperti saat pelatihan model
feature_order = ["Age", "Gender", "TB", "DB", "Alkphos", "SGPT", "SGOT", "TP", "ALB", "A/G Ratio"]
input_df = input_df[feature_order]

# Normalisasi input
input_scaled = scaler.transform(input_df)

# Prediksi
prediction = model.predict(input_scaled)[0]

# Tampilkan hasil
st.subheader("Hasil Prediksi:")
if prediction == 1:
    st.error("Pasien Diprediksi Mengidap Penyakit Liver")
else:
    st.success("Pasien Diprediksi Sehat")
