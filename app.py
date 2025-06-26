import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Judul Aplikasi
st.title("Prediksi Penyakit Liver (ILPD Dataset)")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ilpd.csv", header=None)
    df.columns = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'SGPT', 'SGOT', 'TP', 'ALB', 'A/G Ratio', 'Dataset']
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Dataset'] = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

df = load_data()

# Menampilkan data awal
if st.checkbox("Tampilkan Data Awal"):
    st.write(df.head())

# Sidebar input pengguna
st.sidebar.header("Input Data Pasien")
age = st.sidebar.slider("Usia", 4, 90, 45)
gender = st.sidebar.selectbox("Jenis Kelamin", ['Male', 'Female'])
tb = st.sidebar.slider("Total Bilirubin", 0.1, 75.0, 1.0)
db = st.sidebar.slider("Direct Bilirubin", 0.1, 20.0, 0.5)
alkphos = st.sidebar.slider("Alkaline Phosphotase", 100, 2000, 250)
sgpt = st.sidebar.slider("SGPT", 10, 2000, 30)
sgot = st.sidebar.slider("SGOT", 10, 2000, 35)
tp = st.sidebar.slider("Total Protein", 2.0, 9.0, 6.5)
alb = st.sidebar.slider("Albumin", 0.5, 6.5, 3.0)
ag_ratio = st.sidebar.slider("A/G Ratio", 0.1, 2.5, 1.0)

# Pilih model
model_choice = st.sidebar.selectbox("Pilih Model", ["K-Nearest Neighbors", "Decision Tree", "Naive Bayes"])

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load model sesuai pilihan
if model_choice == "K-Nearest Neighbors":
    with open("model_knn.pkl", "rb") as f:
        model = pickle.load(f)
elif model_choice == "Decision Tree":
    with open("model_dt.pkl", "rb") as f:
        model = pickle.load(f)
elif model_choice == "Naive Bayes":
    with open("model_nb.pkl", "rb") as f:
        model = pickle.load(f)

# Siapkan input user
input_data = np.array([[age, 1 if gender == 'Male' else 0, tb, db, alkphos,
                        sgpt, sgot, tp, alb, ag_ratio]])
input_scaled = scaler.transform(input_data)

# Prediksi
prediction = model.predict(input_scaled)[0]
result = "Pasien Memiliki Penyakit Liver" if prediction == 1 else "Pasien Sehat"

# Tampilkan hasil
st.subheader("Hasil Prediksi:")
st.success(result)
