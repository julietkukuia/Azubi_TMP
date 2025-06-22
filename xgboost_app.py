# Full Streamlit app script with dropdowns and internal one-hot encoding based on user selections
app_code = '''
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

# Load the trained model and feature names
model = joblib.load("xgb_subscription_model.pkl")
feature_order = joblib.load("xgb_model_features.pkl")

st.set_page_config(page_title="Term Deposit Predictor", layout="centered")
st.title("💼 Term Deposit Subscription Predictor")

st.markdown("Provide client details below. The model will predict whether the client is likely to subscribe to a term deposit.")

# Define dropdown categories
job_categories = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                  'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
marital_categories = ['married', 'single', 'divorced', 'unknown']
education_categories = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                        'illiterate', 'professional.course', 'university.degree', 'unknown']
contact_categories = ['telephone', 'cellular']
month_categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_of_week_categories = ['mon', 'tue', 'wed', 'thu', 'fri']
poutcome_categories = ['failure', 'nonexistent', 'success']

# Collect user input
with st.expander("📋 Personal Information"):
    age = st.number_input("Age", 18, 100, 35)
    job = st.selectbox("Job", job_categories)
    marital = st.selectbox("Marital Status", marital_categories)
    education = st.selectbox("Education Level", education_categories)

with st.expander("🏦 Financial Information"):
    default = st.selectbox("Has Credit in Default?", ['no', 'yes']) == 'yes'
    balance = st.number_input("Account Balance", value=0.0)
    housing = st.selectbox("Has Housing Loan?", ['no', 'yes']) == 'yes'
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes']) == 'yes'

with st.expander("📞 Contact Campaign"):
    contact = st.selectbox("Contact Type", contact_categories)
    month = st.selectbox("Last Contact Month", month_categories)
    day_of_week = st.selectbox("Day of the Week", day_of_week_categories)
    campaign = st.slider("Number of Contacts During Campaign", 1, 50, 2)
    pdays = st.selectbox("Days Since Last Contact", [-1, 0, 1, 5, 10, 20, 999])
    previous = st.slider("Previous Contacts", 0, 50, 0)
    poutcome = st.selectbox("Previous Campaign Outcome", poutcome_categories)

with st.expander("📈 Economic Indicators"):
    emp_var_rate = st.number_input("Employment Variation Rate", -5.0, 2.0, 1.1)
    cons_price_idx = st.number_input("Consumer Price Index", 90.0, 100.0, 93.2)
    cons_conf_idx = st.number_input("Consumer Confidence Index", -60.0, -20.0, -40.0)
    euribor3m = st.number_input("Euribor 3-Month Rate", 0.0, 6.0, 4.0)
    nr_employed = st.number_input("Number of Employees", 4800.0, 5400.0, 5191.0)

# Encode inputs into feature_order format
input_data = dict.fromkeys(feature_order, 0)

input_data['age'] = age
input_data['balance'] = balance
input_data['campaign'] = campaign
input_data['pdays'] = pdays
input_data['previous'] = previous
input_data['emp.var.rate'] = emp_var_rate
input_data['cons.price.idx'] = cons_price_idx
input_data['cons.conf.idx'] = cons_conf_idx
input_data['euribor3m'] = euribor3m
input_data['nr.employed'] = nr_employed
input_data['default'] = int(default)
input_data['housing'] = int(housing)
input_data['loan'] = int(loan)

# One-hot encode categorical selections
input_data[f'job_{job}'] = 1
input_data[f'marital_{marital}'] = 1
input_data[f'education_{education}'] = 1
input_data[f'contact_{contact}'] = 1
input_data[f'month_{month}'] = 1
input_data[f'day_of_week_{day_of_week}'] = 1
input_data[f'poutcome_{poutcome}'] = 1

# Reorder and convert to DataFrame
input_df = pd.DataFrame([input_data])[feature_order]

if st.button("Predict Subscription"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    st.write(f"**Prediction:** {'✅ Yes' if prediction == 1 else '❌ No'}")
    st.write(f"**Probability of Subscription:** {probability:.2%}")
'''

# Save to file
app_path = "xgboost_app_clean.py"
with open(app_file_path, "w", encoding="utf-8") as f:
    f.write(app_code_dropdowns)

app_path
