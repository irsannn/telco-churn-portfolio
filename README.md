# Telco Customer Churn Prediction

> End-to-end data science portfolio menggunakan framework CRISP-DM untuk memprediksi pelanggan yang berisiko churn pada industri telekomunikasi.

## Latar Belakang Bisnis

Perusahaan telekomunikasi kehilangan pelanggan (churn) terus-menerus. Biaya akuisisi pelanggan baru 5-7x lebih mahal daripada mempertahankan pelanggan lama. Proyek ini membangun sistem prediksi churn untuk membantu tim marketing mengidentifikasi pelanggan berisiko dan mengambil tindakan retensi lebih awal.

## Tech Stack

| Kategori | Tools |
|---|---|
| Bahasa | Python 3 |
| Data Processing | pandas, numpy |
| Visualisasi | matplotlib, seaborn |
| Machine Learning | scikit-learn, XGBoost |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |

## Struktur Proyek

telco-churn-portfolio/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   └── pipeline.py
├── notebooks/
│   └── 01_eda_analyst.ipynb
├── .gitignore
├── requirements.txt
└── README.md

## Progress CRISP-DM

- [x] Business Understanding
- [x] Data Understanding
- [x] Data Preparation (ETL Pipeline)
- [x] Exploratory Data Analysis
- [ ] Modeling
- [ ] Evaluation
- [ ] Deployment

## Key Insights dari EDA

| # | Insight | Angka Kunci |
|---|---|---|
| 1 | Class Imbalance | 26.5% churn vs 73.5% tidak |
| 2 | Kontrak Month-to-Month | 42.6% churn (15x dari 2-year) |
| 3 | Masa Kritis 10 Bulan | Median tenure churn = 10 bulan |
| 4 | Harga Tinggi | Churners bayar $79.70 vs $64.55 |

Persona pelanggan paling berisiko churn: kontrak month-to-month, berlangganan kurang dari 12 bulan, dan membayar lebih dari $70 per bulan.

## Cara Menjalankan

1. Clone repo ini
   git clone https://github.com/irsannn/telco-churn-portfolio.git
   cd telco-churn-portfolio

2. Setup virtual environment
   python -m venv venv

3. Aktifkan venv
   Windows: venv\Scripts\activate
   Mac/Linux: source venv/bin/activate

4. Install dependencies
   pip install -r requirements.txt

5. Download dataset dari Kaggle
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   Simpan file WA_Fn-UseC_-Telco-Customer-Churn.csv di folder data/raw/

6. Jalankan ETL pipeline
   python src/pipeline.py

   Output: data/processed/telco_churn_clean.csv

7. Buka notebook EDA
   notebooks/01_eda_analyst.ipynb

## Dataset

Sumber: Telco Customer Churn - Kaggle

- 7.043 baris awal, 21 kolom
- Setelah cleaning: 7.010 baris, 20 kolom
- Class imbalance: sekitar 26% pelanggan churn
- Missing value: 11 baris pada kolom TotalCharges (dihapus)

## Author

irsannn

- GitHub: https://github.com/irsannn

## Lisensi

Proyek ini menggunakan dataset publik dari Kaggle. Bebas digunakan untuk tujuan pembelajaran.