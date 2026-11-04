import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    h1 {
        font-size: 42px !important;
        font-weight: 800 !important;
    }

    h2 {
        font-size: 30px !important;
        font-weight: 750 !important;
    }

    h3 {
        font-size: 25px !important;
        font-weight: 700 !important;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Employee Attrition Prediction")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    features = joblib.load("feature_columns.json")
    return model, features

try:
    model, features = load_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.write("")
st.header("Enter Employee Details")

# Two columns
col1, col2 = st.columns(2, gap="medium")

with col1:

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=60000,
        step=1000
    )

    age = st.slider(
        "Age",
        min_value=18,
        max_value=60,
        value=37
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=42,
        value=8
    )

    overtime = st.selectbox(
        "OverTime",
        options=["No", "Yes"],
        index=0
    )

    daily_rate = st.number_input(
        "Daily Rate",
        min_value=0,
        value=800
    )

with col2:

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )

    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=0,
        value=50
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=0,
        max_value=29,
        value=5
    )

    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=0,
        value=20000
    )

    num_companies_worked = st.number_input(
        "Num Companies Worked",
        min_value=0,
        max_value=9,
        value=2
    )

# Input data
input_data = {
    "MonthlyIncome": monthly_income,
    "Age": age,
    "TotalWorkingYears": total_working_years,
    "OverTime": 1 if overtime == "Yes" else 0,
    "DailyRate": daily_rate,
    "YearsAtCompany": years_at_company,
    "HourlyRate": hourly_rate,
    "DistanceFromHome": distance_from_home,
    "MonthlyRate": monthly_rate,
    "NumCompaniesWorked": num_companies_worked
}

# Create dataframe in the exact feature order
input_df = pd.DataFrame([input_data])

# Verify features
missing_features = [
    feature for feature in features
    if feature not in input_df.columns
]

if missing_features:
    st.error(f"Missing features: {missing_features}")
    st.stop()

input_df = input_df[features]

# Input preview
st.write("")
st.subheader("Input Preview")

st.dataframe(
    input_df,
    use_container_width=True,
    hide_index=False
)

st.write("")

# Prediction button
if st.button("🔍 Predict Attrition"):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][
        list(model.classes_).index(1)
    ]

    if prediction == 1:
        st.error(
            f"⚠️ High risk of attrition (Prob: {probability:.2f})"
        )
    else:
        st.success(
            f"✅ Low risk of attrition (Prob: {probability:.2f})"
        )
