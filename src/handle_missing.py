import pandas as pd

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_visa_dataset.csv")

print("Original shape:", df.shape)

# Step 1: Drop rows where processing time cannot be computed
df_model = df.dropna(subset=['processing_time_days_clean'])

print("After dropping missing processing_time rows:", df_model.shape)

# Step 2: Save model-ready dataset
df_model.to_csv("outputs/final_model_ready_dataset.csv", index=False)

print("\nSaved final_model_ready_dataset.csv successfully!")

