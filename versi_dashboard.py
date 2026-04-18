import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# KONFIGURASI PAGE
# =========================
st.set_page_config(
    page_title="CKD Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
feature_names = joblib.load("feature_names.pkl")
encoders = joblib.load("encoders.pkl")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧠 CKD System")

st.sidebar.markdown("### 📊 Info Model")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("Feature Selection: ANOVA")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset Loaded")

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi prediksi penyakit ginjal kronis (CKD)")

# =========================
# HEADER
# =========================
st.title("🩺 Dashboard Prediksi CKD")
st.markdown("Sistem pendukung keputusan berbasis Machine Learning")

# =========================
# LAYOUT 3 KOLOM
# =========================
col1, col2, col3 = st.columns([2, 2, 1.5])

# =========================
# FORM INPUT (KIRI)
# =========================
with col1:
    st.subheader("📋 Input Data Pasien")

    input_data = {}

    for feature in feature_names:
        if feature in encoders:
            options = encoders[feature].classes_
            val = st.selectbox(feature, options, key=feature)
            val = encoders[feature].transform([val])[0]
        else:
            val = st.number_input(feature, value=0.0, key=feature)

        input_data[feature] = val

# =========================
# PROSES PREDIKSI
# =========================
predict_clicked = st.button("🔍 Prediksi Sekarang")

# =========================
# HASIL (TENGAH)
# =========================
with col2:
    st.subheader("📊 Hasil Diagnosis")

    if predict_clicked:
        df_input = pd.DataFrame([input_data])

        X_scaled = scaler.transform(df_input)
        X_sel = selector.transform(X_scaled)

        pred = model.predict(X_sel)[0]
        prob = model.predict_proba(X_sel)[0][1]

        if pred == 1:
            st.error("⚠️ POSITIF CKD")
        else:
            st.success("✅ NEGATIF CKD")

        st.metric(
            label="Probabilitas CKD",
            value=f"{prob*100:.2f}%"
        )

        # Interpretasi sederhana
        if prob > 0.7:
            st.warning("Risiko Tinggi")
        elif prob > 0.4:
            st.info("Risiko Sedang")
        else:
            st.success("Risiko Rendah")

# =========================
# VISUAL RISK (KANAN)
# =========================
with col3:
    st.subheader("📈 Risk Level")

    if predict_clicked:
        risk = prob

        st.progress(int(risk * 100))

        st.write("### Indikator Risiko")

        if risk > 0.7:
            st.markdown("🔴 **Tinggi**")
        elif risk > 0.4:
            st.markdown("🟠 **Sedang**")
        else:
            st.markdown("🟢 **Rendah**")

# =========================
# FOOTER ANALYTICS
# =========================
st.markdown("---")
st.subheader("📌 Ringkasan Input")

if predict_clicked:
    st.dataframe(pd.DataFrame([input_data]))
