"""
Dashboard Streamlit untuk Telco Customer Churn Prediction.
Author: irsannn
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD MODEL & ARTEFAK
# ============================================================
@st.cache_resource
def load_artifacts():
    base_path = Path(__file__).parent.parent
    model = joblib.load(base_path / "models" / "rf_churn_model.pkl")
    scaler = joblib.load(base_path / "models" / "scaler.pkl")
    with open(base_path / "models" / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


model, scaler, feature_names = load_artifacts()

# ============================================================
# HEADER
# ============================================================
st.title("📊 Telco Customer Churn Predictor")
st.markdown("""
Dashboard interaktif untuk memprediksi **kemungkinan churn** pelanggan telekomunikasi.

**Model:** Random Forest (threshold 0.4) — Recall ~80%, ROC-AUC 0.84
""")

st.divider()

# ============================================================
# SIDEBAR — INPUT PELANGGAN
# ============================================================
st.sidebar.header("👤 Input Data Pelanggan")

tenure = st.sidebar.slider("Tenure (bulan)", 0, 72, 60)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
total_charges = st.sidebar.number_input(
    "Total Charges ($)", 0.0, 10000.0,
    value=float(tenure * monthly_charges)
)

contract = st.sidebar.selectbox(
    "Jenis Kontrak",
    ["Month-to-month", "One year", "Two year"],
    index=2
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Metode Pembayaran",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)",
     "Credit card (automatic)"],
    index=2
)

gender = st.sidebar.radio("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.radio("Senior Citizen", ["No", "Yes"])


# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict_churn(input_dict):
    """Predict churn probability."""
    df_input = pd.DataFrame([input_dict])

    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]

    # Tambah 1 baris dummy agar get_dummies bisa encode
    dummy_row = {}
    for col in df_input.columns:
        if col in categorical_cols:
            if col == "Contract":
                dummy_row[col] = "Two year" if input_dict[col] != "Two year" else "Month-to-month"
            elif col == "InternetService":
                dummy_row[col] = "DSL" if input_dict[col] != "DSL" else "Fiber optic"
            elif col == "PaymentMethod":
                dummy_row[col] = "Mailed check" if input_dict[col] != "Mailed check" else "Electronic check"
            elif col == "gender":
                dummy_row[col] = "Female" if input_dict[col] != "Female" else "Male"
            else:
                dummy_row[col] = "Yes" if input_dict[col] != "Yes" else "No"
        else:
            dummy_row[col] = input_dict[col]

    df_input = pd.concat([df_input, pd.DataFrame([dummy_row])], ignore_index=True)

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=False)
    df_encoded = df_encoded.astype(float)
    df_encoded = df_encoded.iloc[[0]]

    # Buat DataFrame kosong sesuai feature_names
    df_final = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Isi kolom yang match
    for col in df_encoded.columns:
        if col in df_final.columns:
            df_final[col] = df_encoded[col].values

    # Scaling & prediksi
    df_scaled = scaler.transform(df_final)
    proba = model.predict_proba(df_scaled)[0, 1]

    return proba


# ============================================================
# TOMBOL PREDIKSI
# ============================================================
predict_button = st.sidebar.button("🔮 Prediksi Churn", type="primary")

# ============================================================
# KONTEN UTAMA
# ============================================================
if predict_button:
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PaperlessBilling": "Yes",
    }

    proba = predict_churn(input_dict)
    prediksi = "CHURN" if proba >= 0.4 else "TIDAK CHURN"

    st.subheader("🎯 Hasil Prediksi")

    col1, col2 = st.columns([1, 1])

    with col1:
        if proba >= 0.4:
            st.error("⚠️ **Pelanggan berisiko CHURN**")
        else:
            st.success("✅ **Pelanggan cenderung TIDAK churn**")

        st.metric("Probabilitas Churn", f"{proba*100:.1f}%")
        st.metric("Threshold", "40%")
        st.metric("Status", prediksi)

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Probabilitas Churn (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if proba >= 0.4 else "#2ecc71"},
                "steps": [
                    {"range": [0, 40], "color": "#d5f4e6"},
                    {"range": [40, 70], "color": "#fef9e7"},
                    {"range": [70, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 40
                }
            }
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("💡 Rekomendasi Tindakan")

    if proba >= 0.4:
        st.markdown("""
        **Pelanggan ini berisiko churn.** Rekomendasi:
        1. 🎁 Tawarkan **diskon atau promo retensi** khusus
        2. 📞 Hubungi lewat **customer service** untuk cek kepuasan
        3. 📝 Dorong **upgrade ke kontrak jangka panjang**
        4. 🎯 Tawarkan **layanan tambahan** dengan harga khusus
        """)
    else:
        st.markdown("""
        **Pelanggan ini cenderung setia.** Rekomendasi:
        1. ✅ Pertahankan **kualitas layanan**
        2. 📧 Kirim **newsletter atau info produk baru**
        3. 🎁 Berikan **loyalty reward**
        4. 📊 Pantau secara berkala
        """)

else:
    st.info("👈 Isi data pelanggan di sidebar kiri, lalu klik **🔮 Prediksi Churn**.")
    st.divider()
    st.subheader("📈 Insight dari Data Historis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Churn Rate", "26.5%", "dari 7.010 pelanggan")
    with col2:
        st.metric("Churn Month-to-month", "42.6%", "vs 2.8% two-year")
    with col3:
        st.metric("Median Tenure Churn", "10 bulan", "vs 38 bulan loyal")