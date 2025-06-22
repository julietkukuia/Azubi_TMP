
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

# Load XGBoost model and feature list
model = joblib.load('xgb_subscription_model.pkl')
feature_order = joblib.load('xgb_model_features.pkl')

st.set_page_config(page_title="Subscription Predictor", layout="centered")
st.title("💡 Term Deposit Subscription Predictor")
st.markdown("Fill in the client information below to predict subscription outcome.")

# Input UI
user_inputs = {}
for feature in feature_order:
    if 'age' in feature:
        user_inputs[feature] = st.number_input('Age', min_value=18, max_value=100, value=35, key=feature)
    elif 'campaign' in feature:
        user_inputs[feature] = st.slider('Campaign Contacts', 1, 50, 2, key=feature)
    elif 'pdays' in feature:
        user_inputs[feature] = st.selectbox('Days Since Last Contact', [-1, 0, 1, 5, 10, 20, 999], key=feature)
    elif 'previous' in feature:
        user_inputs[feature] = st.slider('Previous Contacts', 0, 50, 0, key=feature)
    elif 'euribor3m' in feature:
        user_inputs[feature] = st.number_input('Euribor 3-month Rate', 0.0, 6.0, 4.0, key=feature)
    elif 'emp.var.rate' in feature:
        user_inputs[feature] = st.number_input('Employment Variation Rate', -5.0, 2.0, 1.1, key=feature)
    elif feature in ['default', 'housing', 'loan']:
        choice = st.selectbox(f"Has {feature.capitalize()}?", ['yes', 'no'], key=feature)
        user_inputs[feature] = 1 if choice == 'yes' else 0
    elif 'contact' in feature:
        contact_type = st.selectbox('Contact Type', ['telephone', 'cellular'], key=feature)
        user_inputs[feature] = 1 if contact_type == 'cellular' else 0
    elif any(categorical in feature for categorical in ['job_', 'marital_', 'education_', 'month_', 'day_of_week_', 'poutcome_']):
        user_inputs[feature] = st.checkbox(f"{feature}", key=feature)
        user_inputs[feature] = 1 if user_inputs[feature] else 0
    else:
        user_inputs[feature] = st.number_input(feature, value=0.0, key=feature)

# Convert to DataFrame
input_df = pd.DataFrame([user_inputs])[feature_order]

# Prediction
if st.button("Predict Subscription"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader("📊 Prediction Result")
    st.write(f"**Prediction:** {'✅ Yes' if prediction == 1 else '❌ No'}")
    st.write(f"**Subscription Probability:** {probability:.2%}")
