# src/data_load.py
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "visa_dataset_1000.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # try parse date columns if strings
    for col in ["submission_date", "decision_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(5))
    print("\nMissing per column:\n", df.isnull().sum())
