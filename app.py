import streamlit as st
import joblib
import pandas as pd

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI Visa Processing Time Estimator",
    page_icon="🛂",
    layout="centered"
)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return joblib.load("notebooks/models/visa_processing_time_model.pkl")

model = load_model()

# -------------------- App Title --------------------
st.title("🛂 AI Visa Processing Time Estimator")
st.write("Predict visa processing time using Machine Learning")

st.divider()

# -------------------- User Inputs --------------------
st.subheader("Enter Application Details")

application_year = st.selectbox(
    "Application Year",
    [2021, 2022, 2023, 2024],
    key="year"
)

submission_month = st.selectbox(
    "Submission Month",
    list(range(1, 13)),
    key="month"
)

country = st.selectbox(
    "Applicant Country",
    [
        "India", "China", "USA", "Canada", "UK",
        "Germany", "France", "Australia"
    ],
    key="country"
)

season = st.selectbox(
    "Season",
    ["Winter", "Spring", "Summer", "Autumn"],
    key="season"
)

st.divider()

# -------------------- Feature Preparation --------------------
input_data = dict.fromkeys(model.feature_names_in_, 0)

input_data["application_year"] = application_year
input_data["submission_month"] = submission_month
input_data["submission_dayofweek"] = 3
input_data["submission_quarter"] = (submission_month - 1) // 3 + 1

country_col = f"applicant_country_{country}"
if country_col in input_data:
    input_data[country_col] = 1

season_col = f"season_{season}"
if season_col in input_data:
    input_data[season_col] = 1

input_df = pd.DataFrame([input_data])

# -------------------- Prediction --------------------
if st.button("Predict Processing Time", key="predict"):
    prediction = model.predict(input_df)[0]
    prediction = max(1, int(prediction))

    st.success(f"⏳ Estimated Processing Time: **{prediction} days**")

    lower = max(1, int(prediction * 0.9))
    upper = int(prediction * 1.1)

    st.info(f"📌 Expected Range: **{lower} – {upper} days**")
