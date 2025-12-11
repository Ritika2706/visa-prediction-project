# src/preprocess.py
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "visa_dataset_1000.csv"
OUTPUT = ROOT / "outputs" / "cleaned_visa_dataset.csv"

def compute_processing_days(df):
    # ensure date columns are datetime
    df['submission_date'] = pd.to_datetime(df.get('submission_date'), errors='coerce')
    df['decision_date'] = pd.to_datetime(df.get('decision_date'), errors='coerce')
    # compute days diff (NaT will produce NaN)
    df['processing_days_computed'] = (df['decision_date'] - df['submission_date']).dt.days
    return df

def reconcile_processing_column(df):
    # If provided processing_time_days exists, compare and prefer computed if available
    if 'processing_time_days' in df.columns:
        # if computed is not null, use it; else use existing column
        df['processing_time_days_clean'] = df['processing_days_computed'].where(
            df['processing_days_computed'].notnull(),
            df['processing_time_days']
        )
    else:
        df['processing_time_days_clean'] = df['processing_days_computed']
    return df

def basic_cleaning(df):
    # Drop exact duplicates
    df = df.drop_duplicates(subset='application_id', keep='first') if 'application_id' in df.columns else df.drop_duplicates()
    # Strip whitespace in object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    return df

def handle_missing_values(df):
    # Example strategies:
    # - If submission_date or decision_date missing -> processing time cannot be computed -> keep as NaN
    # - For categorical columns with missing, fill with "Unknown"
    for cat in ['applicant_country', 'visa_type', 'processing_center', 'decision', 'season']:
        if cat in df.columns:
            df[cat] = df[cat].fillna('Unknown')
    # For numeric missing processing_time_days_clean -> leave NaN (we'll drop or impute later before modeling)
    return df

def cap_outliers(df, col='processing_time_days_clean', lower=0, upper_percentile=0.99):
    if col in df.columns:
        # remove negative values (invalid)
        df.loc[df[col] < 0, col] = pd.NA
        # cap extremely large using percentile
        upper_val = df[col].quantile(upper_percentile)
        df[col] = df[col].clip(upper=upper_val)
    return df

def save_clean(df, path=OUTPUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned dataset to {path}")

if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    df = compute_processing_days(df)
    df = reconcile_processing_column(df)
    df = basic_cleaning(df)
    df = handle_missing_values(df)
    df = cap_outliers(df)
    save_clean(df)
