import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


st.set_page_config(page_title="Prediksi Kepadatan", layout="centered")
st.title("Prediksi Kepadatan Penduduk")

# Load data
df = pd.read_csv('data_final.csv')

def kategori_kepadatan(nilai):
    if nilai < 500:
        return "Rendah 🟢"
    elif 500 <= nilai <= 1500:
        return "Sedang 🟡"
    else:
        return "Tinggi 🔴"

# Buat kolom kategori
df['Kategori_Kepadatan'] = df['Kepadatan Penduduk per km persegi Km2'].apply(kategori_kepadatan)

# Sidebar untuk input
st.sidebar.header("Input Data")

# Input fields
jumlah_penduduk = st.sidebar.number_input("Jumlah Penduduk (Ribu)", 0.0, 500.0, 50.0, 10.0)
persentase_penduduk = st.sidebar.number_input("Persentase Penduduk (%)", 0.0, 100.0, 5.0, 1.0)
laju_pertumbuhan = st.sidebar.number_input("Laju Pertumbuhan (%)", 0.0, 10.0, 0.68, 0.1)
rasio_jenis_kelamin = st.sidebar.number_input("Rasio Jenis Kelamin", 0.0, 200.0, 103.0, 1.0)
luas_wilayah = st.sidebar.number_input("Luas Wilayah (Km²)", 0.0, 500.0, 100.0, 10.0)

# Tombol prediksi
if st.sidebar.button("Prediksi"):
    # Preprocessing dan model (sama seperti notebook)
    features = ['Jumlah Penduduk (Ribu)', 'Persentase Penduduk', 
                'Laju Pertumbuhan Penduduk per Tahun',
                'Rasio Jenis Kelamin Penduduk', 'Luas Wilayah (Km2)']
    
    X = df[features]
    y = df['Kepadatan Penduduk per km persegi Km2']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediksi test set untuk evaluasi
    y_pred = model.predict(X_test)
    
    # Hitung MSE dan R2 (dengan code Anda)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Prediksi dengan input user
    input_data = np.array([[
        jumlah_penduduk,
        persentase_penduduk,
        laju_pertumbuhan,
        rasio_jenis_kelamin,
        luas_wilayah
    ]])
    
    input_data_scaled = scaler.transform(input_data)
    predicted_density = model.predict(input_data_scaled)[0]
    
    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kepadatan Diprediksi", f"{predicted_density:,.0f} orang/km²")
    with col2:
        kategori = kategori_kepadatan(predicted_density)
        st.metric("Kategori", kategori)
    
    # Tampilkan MSE dan R2
    st.subheader("Evaluasi Model")
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'R-squared (R2): {r2}')
    
    # Visualisasi
    st.subheader("Visualisasi")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot hasil prediksi vs data asli
    ax.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Prediksi vs Aktual')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Garis Ideal')
    
    ax.set_xlabel('Kepadatan Aktual (orang/km²)')
    ax.set_ylabel('Kepadatan Prediksi (orang/km²)')
    ax.set_title('Perbandingan Prediksi vs Aktual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# Tampilkan data asli
st.sidebar.markdown("---")
if st.sidebar.checkbox("Tampilkan Data"):
    st.subheader("Data Asli")
    st.dataframe(df[['Kecamatan', 'Kepadatan Penduduk per km persegi Km2', 'Kategori_Kepadatan']].head(10))

# Footer
st.markdown("---")
st.caption("Aplikasi Prediksi Sederhana Kel 5| Model Linear Regression")