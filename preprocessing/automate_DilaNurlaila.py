"""
automate_Nama-siswa.py

Script otomatisasi preprocessing dataset Pima Indians Diabetes.
Konversi dari tahapan manual pada notebook Eksperimen_SML_Nama-siswa.ipynb.

Fungsi utama: preprocess_data(input_path, output_dir)
Mengembalikan data yang sudah siap dilatih (train.csv, test.csv).
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLS_NO_ZERO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(input_path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    df = pd.read_csv(input_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Mengganti nilai 0 yang tidak masuk akal secara medis dengan median kolom."""
    df_clean = df.copy()
    for col in COLS_NO_ZERO:
        median_val = df_clean.loc[df_clean[col] != 0, col].median()
        df_clean[col] = df_clean[col].replace(0, median_val)
    return df_clean


def split_and_scale(df_clean: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data menjadi train/test lalu melakukan standarisasi fitur."""
    X = df_clean.drop(columns=["Outcome"])
    y = df_clean["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    train_out = X_train_scaled.copy()
    train_out["Outcome"] = y_train.values
    test_out = X_test_scaled.copy()
    test_out["Outcome"] = y_test.values

    return train_out, test_out


def preprocess_data(input_path: str, output_dir: str) -> None:
    """Pipeline lengkap: load -> clean -> split & scale -> save."""
    df = load_data(input_path)
    df_clean = clean_data(df)
    train_out, test_out = split_and_scale(df_clean)

    os.makedirs(output_dir, exist_ok=True)
    train_out.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_out.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Preprocessing selesai. Data tersimpan di: {output_dir}")
    print(f"  - train.csv: {train_out.shape}")
    print(f"  - test.csv : {test_out.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otomatisasi preprocessing dataset diabetes.")
    parser.add_argument(
        "--input", type=str, default="../diabetes_raw/diabetes.csv",
        help="Path ke file dataset mentah (CSV)."
    )
    parser.add_argument(
        "--output", type=str, default="../diabetes_preprocessing",
        help="Folder output untuk menyimpan hasil preprocessing."
    )
    args = parser.parse_args()

    preprocess_data(args.input, args.output)
