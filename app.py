import streamlit as st
import pandas as pd
import pickle
import os

MODEL_DIR = 'ModelUAS'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_heart_disease_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Muat model dan scaler yang telah disimpan
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    with open(SCALER_PATH, 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error(
        f"File model atau scaler tidak ditemukan. Pastikan folder '{MODEL_DIR}' dengan file .pkl ada di direktori yang sama dengan app.py.")
    st.stop()


# Judul Aplikasi
st.title('Heart Disease Prediction App')
st.markdown("Aplikasi ini memprediksi kemungkinan seseorang menderita penyakit jantung berdasarkan 13 atribut klinis.")

# Membuat kolom untuk input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Usia (tahun)', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Jenis Kelamin', (0, 1), format_func=lambda x: 'Perempuan' if x == 0 else 'Laki-laki')
    cp = st.selectbox('Tipe Nyeri Dada (cp)', (0, 1, 2, 3), help="0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic")
    trestbps = st.number_input('Tekanan Darah Istirahat (mm Hg)', min_value=50, max_value=250, value=120)
    chol = st.number_input('Kolesterol Serum (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Gula Darah Puasa > 120 mg/dl', (0, 1), format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

with col2:
    restecg = st.selectbox('Hasil EKG Istirahat (restecg)', (0, 1, 2), help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
    thalach = st.number_input('Denyut Jantung Maksimum (thalach)', min_value=50, max_value=220, value=150)
    exang = st.selectbox('Angina Akibat Olahraga (exang)', (0, 1), format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
    oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope', (0, 1, 2), help="0: Upsloping, 1: Flat, 2: Downsloping")
    ca = st.selectbox('Jumlah Pembuluh Darah Utama (ca)', (0, 1, 2, 3))
    thal = st.selectbox('Thal', (1, 2, 3), help="1: Normal, 2: Fixed Defect, 3: Reversible Defect")

# Tombol untuk prediksi
if st.button('Prediksi'):
    # Membuat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
        'chol': [chol], 'fbs': [fbs], 'restecg': [restecg],
        'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak],
        'slope': [slope], 'ca': [ca], 'thal': [thal]
    })
    
    # Pastikan urutan kolom sama dengan saat training
    feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data = input_data[feature_order]

    # Scaling input data menggunakan scaler yang sudah di-fit
    input_data_scaled = scaler.transform(input_data)
    
    # Melakukan prediksi
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    st.subheader('Hasil Prediksi:')
    if prediction[0] == 1:
        st.error(f'Pasien terindikasi memiliki Penyakit Jantung (Probabilitas: {prediction_proba[0][1]*100:.2f}%)')
    else:
        st.success(f'Pasien tidak terindikasi memiliki Penyakit Jantung (Probabilitas: {prediction_proba[0][0]*100:.2f}%)')