import pandas as pd

df = pd.read_csv("outputs/cleaned_visa_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nMissing Values per Column:\n", df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Show percentage of missing values
missing_percent = df.isnull().mean()*100
print(missing_percent)

missing_percent.plot(kind='bar', figsize=(10,4))
plt.title("Missing Data Percentage per Feature")
plt.show()
