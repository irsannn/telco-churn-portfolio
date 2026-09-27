"""
Pipeline ETL untuk dataset Telco Customer Churn.
Mengubah data mentah (raw) menjadi data bersih (processed).
"""

import pandas as pd
from pathlib import Path
import logging

# Setup logging agar proses terlacak
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Path
RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_PATH = Path("data/processed/telco_churn_clean.csv")


def extract(path: Path) -> pd.DataFrame:
    """Membaca data mentah dari CSV."""
    logging.info(f"Membaca data dari {path}")
    df = pd.read_csv(path)
    logging.info(f"Data berhasil dibaca: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan dan mentransformasi data."""

    # 1. Hapus kolom customerID (tidak relevan untuk modeling)
    df = df.drop(columns=["customerID"])
    logging.info("Kolom 'customerID' dihapus")

    # 2. Konversi TotalCharges dari object ke numerik
    #    Nilai kosong (' ') akan menjadi NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logging.info("Kolom 'TotalCharges' dikonversi ke numerik")

    # 3. Tangani missing value di TotalCharges
    #    Hanya ~11 baris, jadi drop (alasan: <0.2% data)
    missing_before = df["TotalCharges"].isna().sum()
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    logging.info(f"{missing_before} baris dengan TotalCharges kosong dihapus")

    # 4. Standarisasi nilai kategori (opsional)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    logging.info("Kolom 'Churn' diubah menjadi 0/1")

    # 5. Cek duplikat
    duplikat = df.duplicated().sum()
    if duplikat > 0:
        df = df.drop_duplicates()
        logging.info(f"{duplikat} baris duplikat dihapus")
    else:
        logging.info("Tidak ada baris duplikat")

    return df


def load(df: pd.DataFrame, path: Path) -> None:
    """Menyimpan data bersih ke folder processed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Data bersih disimpan di {path}")


def main():
    logging.info("=== PIPELINE DIMULAI ===")
    df_raw = extract(RAW_PATH)
    df_clean = transform(df_raw)
    load(df_clean, PROCESSED_PATH)
    logging.info("=== PIPELINE SELESAI ===")
    logging.info(f"Shape akhir: {df_clean.shape}")


if __name__ == "__main__":
    main()