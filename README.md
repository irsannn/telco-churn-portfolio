# Telco Customer Churn Prediction

End-to-end data science portfolio menggunakan framework **CRISP-DM** untuk memprediksi churn pelanggan telekomunikasi.

🔗 **Live Demo:** [telco-churn-dashboard.streamlit.app](https://telco-churn-dashboard.streamlit.app)

---

## 📌 Latar Belakang

Biaya akuisisi pelanggan baru **5–7x lebih mahal** daripada mempertahankan pelanggan lama. Proyek ini membangun sistem prediksi churn untuk membantu tim marketing mengidentifikasi pelanggan berisiko.

---

## 🛠️ Tech Stack

| Kategori | Tools |
|---|---|
| Bahasa | Python 3 |
| Data | pandas, numpy |
| Visualisasi | matplotlib, seaborn, plotly |
| ML | scikit-learn, XGBoost |
| Dashboard | Streamlit |

---

## 🗂️ Struktur

telco-churn-portfolio/
├── data/ # raw & processed
├── src/pipeline.py # ETL
├── notebooks/
│ ├── 01_eda_analyst.ipynb
│ └── 02_modeling_scientist.ipynb
├── models/ # model & scaler
├── dashboard/app.py # Streamlit app
└── requirements.txt




---

## 📊 Progress CRISP-DM

- [x] Business Understanding
- [x] Data Understanding (EDA)
- [x] Data Preparation (ETL)
- [x] Modeling (LR, RF, XGBoost)
- [x] Evaluation
- [x] Deployment (Streamlit)

---

## 📈 Key Insights

| # | Insight | Angka |
|---|---|---|
| 1 | Class imbalance | 26.5% churn |
| 2 | Kontrak month-to-month | 42.6% churn (vs 2.8% two-year) |
| 3 | Masa kritis 10 bulan | Median tenure churn = 10 bulan |
| 4 | Harga tinggi | Churners bayar $79.70 (vs $64.55) |

**Persona berisiko:** month-to-month + tenure <12 bulan + bayar >$70/bulan.

---

## 🤖 Hasil Modeling

| Model | Threshold | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.5 | 0.7763 | 0.6154 |
| **Random Forest** | **0.4** | **0.8005** | **0.6279** |
| XGBoost | 0.4 | 0.8248 | 0.6163 |

**Model terpilih:** Random Forest (threshold 0.4) — F1 tertinggi, Recall 80%.

**Top 5 fitur:** tenure, TotalCharges, MonthlyCharges, Contract_Two year, InternetService_Fiber optic.

---

## 🚀 Cara Menjalankan

```bash
# 1. Clone & setup
git clone https://github.com/irsannn/telco-churn-portfolio.git
cd telco-churn-portfolio
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Download dataset dari Kaggle → simpan di data/raw/

# 3. Jalankan pipeline
python src/pipeline.py

# 4. Jalankan dashboard
streamlit run dashboard/app.py



 Dataset
Sumber: Telco Customer Churn — Kaggle

7.043 baris awal → 7.010 setelah cleaning

20 kolom, ~26% churn (imbalanced)

👤 Author
Irsan Khomis

GitHub: @irsannn

text

---

## 🎯 Langkah

1. Buka `README.md`, **`Ctrl+A`** → **`Delete`**.
2. **Paste** kode di atas.
3. **Ganti** `[Nama Kamu]` dan link LinkedIn.
4. Simpan (`Ctrl+S`).
5. Commit & push:
   ```bash
   git add README.md
   git commit -m "docs: simplify README"
   git push



