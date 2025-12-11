import pandas as pd

df = pd.read_csv("outputs/final/final_clean_dataset.csv")

print("\n================ Dataset Validation Report ================\n")
print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}\n")

# ------------------- Missing Values Report -------------------
print("Missing Values:\n", df.isnull().sum())

# ------------------- Sanity Checks ---------------------------
print("\nRunning sanity assertions...")

# 1. Check no missing values remain
assert df.isnull().sum().sum() == 0, "Dataset still contains missing values!"

# 2. Check numeric ranges
assert df['processing_time_days_clean'].min() >= 0, "Negative processing time detected!"

# 3. Identify one-hot encoded binary columns
binary_columns = [col for col in df.columns if 
                  col.startswith("applicant_country_") or
                  col.startswith("visa_type_") or
                  col.startswith("processing_center_") or
                  col.startswith("season_")]

# 4. Validate binary columns only
for col in binary_columns:
    assert set(df[col].unique()).issubset({0,1}), f"{col} has non-binary values"

# 5. Check decision field correctness
assert set(df['decision_binary'].unique()).issubset({0,1}), "decision_binary invalid"

print("\nAll validation checks passed successfully! 🎉")
print("Milestone-1 data is clean and ready for model training.")
