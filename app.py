import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Judul
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

# Sidebar input
st.sidebar.header("Input Data Pasien")

# Input dinamis
input_dict = {}
for column in df.columns[:-1]:  # Kecuali 'Dataset'
    if column == 'Gender':
        input_dict[column] = st.sidebar.selectbox("Jenis Kelamin", ['Male', 'Female'])
    else:
        min_val = float(df[column].min())
        max_val = float(df[column].max())
        mean_val = float(df[column].mean())
        step = (max_val - min_val) / 100.0 if max_val != min_val else 0.01
        input_dict[column] = st.sidebar.slider(column, min_val, max_val, mean_val, step=step)

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

# Siapkan data input
input_df = pd.DataFrame([input_dict])
input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
input_scaled = scaler.transform(input_df)

# Prediksi
prediction = model.predict(input_scaled)[0]
result = "Pasien Memiliki Penyakit Liver" if prediction == 1 else "Pasien Sehat"

# Tampilkan hasil
st.subheader("Hasil Prediksi:")
st.success(result)
