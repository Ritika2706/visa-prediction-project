# AI Enabled Visa Status Prediction and Processing Time Estimator

## 📌 Project Overview
Visa applicants often face uncertainty regarding application status and processing time.  
This project uses machine learning to predict:
- Visa approval status
- Estimated visa processing time

based on historical data such as applicant country, visa type, and application season.

---
## 📅 Week 1: Data Collection & Understanding
Objective:

To understand the problem statement and gather relevant visa application data required for analysis and modeling.

Activities Performed:

Studied visa processing workflows and common influencing factors.

Collected historical / synthetic visa application data.

Identified important attributes such as:

Applicant country

Application year

Submission date

Decision date

Visa decision status

Key Outcomes:

Raw dataset collected.

Clear understanding of features and target variable.

Defined project scope and expected outputs.

Deliverables:

Initial raw dataset

Problem understanding documentation

## 📅 Week 2: Data Cleaning & Preprocessing
Objective:

To clean the collected data and prepare it for analysis and modeling.

Activities Performed:

Removed duplicates and irrelevant columns.

Handled missing values.

Converted date columns into usable formats.

Created target variable:

Visa processing time (in days).

Encoded categorical variables where required.

Key Outcomes:

Clean and structured dataset ready for EDA.

Target variable successfully created.

Deliverables:

final_clean_dataset.csv

## 📅 Week 3: Exploratory Data Analysis (EDA)
Objective:

To analyze the dataset and uncover trends and patterns affecting visa processing time.

Activities Performed:

Performed statistical analysis using pandas.

Created visualizations using Matplotlib and Seaborn.

Analyzed:

Processing time distribution

Country-wise processing trends

Seasonal effects on processing time

Key Outcomes:

Identified major trends and outliers.

Understood how different features affect processing time.

Deliverables:

EDA notebook (eda_week3_4.ipynb)

Visualizations saved in outputs/figures/

## 📅 Week 4: Feature Engineering
Objective:

To enhance model performance by creating meaningful features.

Activities Performed:

Extracted:

Submission month

Submission quarter

Day of week

Created seasonal features (Winter, Summer, etc.).

One-hot encoded categorical variables.

Verified dataset consistency.

Key Outcomes:

Feature-engineered dataset ready for modeling.

Deliverables:

week4_feature_engineered_dataset.csv

## 📅 Week 5: Baseline Model Development
Objective:

To build initial machine learning models for visa processing time prediction.

Activities Performed:

Split data into training and testing sets.

Trained baseline models:

Linear Regression

Random Forest Regressor

Evaluated models using:

MAE

RMSE

R² Score

Key Outcomes:

Random Forest performed better than baseline models.

Deliverables:

Modeling notebook (modeling_week5_6.ipynb)

## 📅 Week 6: Model Optimization & Saving
Objective:

To finalize the best model and prepare it for deployment.

Activities Performed:

Tuned hyperparameters for Random Forest.

Validated model performance.

Saved the final trained model using joblib.

Key Outcomes:

Stable and accurate prediction model finalized.

Deliverables:

best_model.pkl

Final model evaluation results

## 📅 Week 7: Streamlit Web App Development
Objective:

To build a user-friendly web interface for prediction.

Activities Performed:

Designed UI using Streamlit.

Added input fields:

Application year

Submission month

Applicant country

Season

Connected UI with trained ML model.

Ensured feature alignment with training data.

Key Outcomes:

Functional visa processing time estimator.

Deliverables:

app.py

Streamlit UI implementation

## 📅 Week 8: Testing, Deployment & Finalization
Objective:

To test the application and prepare it for final submission.

Activities Performed:

Performed multiple test predictions.

Handled edge cases and errors.

Improved UI messages and result display.

Prepared documentation and GitHub repository.

Key Outcomes:

End-to-end working AI system.

Ready for academic / internship evaluation.

Deliverables:

Fully working Streamlit app

Updated GitHub repository

Final documentation

## 🚀 Final Project Status

✅ Data Collection – Completed
✅ EDA & Feature Engineering – Completed
✅ Model Training – Completed
✅ Web App Development – Completed
✅ Testing & Documentation – Completed

---
## 🚀 Features
- Visa status prediction (Approved / Rejected / Pending)
- Processing time estimation (in days)
- User-friendly interface
- Data-driven insights

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Git & GitHub

---


## ▶️ How to Run the Project
```bash
pip install -r requirements.txt
streamlit run app.py

## 🌐 Deployed Application
You can access the live app here:  
[Live App] https://visa-prediction-project-ersgkhjvxvqxhkr6ugy5wp.streamlit.app/
