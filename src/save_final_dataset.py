import pandas as pd
import os

# Load validated dataset
df = pd.read_csv("G:/visa-prediction-project/outputs/model_validated_dataset.csv")

# Ensure output folder exists
os.makedirs("G:/visa-prediction-project/outputs/final", exist_ok=True)

# Save final dataset for modeling
output_path = "G:/visa-prediction-project/outputs/final/final_clean_dataset.csv"
df.to_csv(output_path, index=False)

print(f"✅ Final dataset saved successfully at:\n{output_path}")
print("Rows:", df.shape[0], " | Columns:", df.shape[1])
