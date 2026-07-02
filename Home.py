import streamlit as st
import pandas as pd
import pickle


st.title("Customer Churn Prediction")
#st.sidebar.title('churn')

account_length = st.number_input("Account Length", min_value=0)
Area_code=st.number_input("Area Code", min_value=0)
International_plan=st.selectbox("International plan",[" Yes","No"])
voice_mail_plan = st.selectbox("Voice Mail Plan", ["Yes", "No"])

number_vmail_messages = st.number_input("Number of Voice Mail Messages", min_value=0)

total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0)

total_day_calls = st.number_input("Total Day Calls", min_value=0)

total_day_charge = st.number_input("Total Day Charge", min_value=0.0)

total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0)

total_eve_calls = st.number_input("Total Evening Calls", min_value=0)

total_eve_charge = st.number_input("Total Evening Charge", min_value=0.0)

total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0)

total_night_calls = st.number_input("Total Night Calls", min_value=0)

total_night_charge = st.number_input("Total Night Charge", min_value=0.0)

total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0)

total_intl_calls = st.number_input("Total International Calls", min_value=0)

total_intl_charge = st.number_input("Total International Charge", min_value=0.0)

customer_service_calls = st.number_input("Customer Service Calls", min_value=0)

if st.button("Predict"):

    # Convert Yes/No to 1/0
    international_plan = 1 if International_plan == "Yes" else 0
    voice_mail_plan = 1 if voice_mail_plan == "Yes" else 0

    # Create feature list
    features = [
        account_length,
        Area_code,
        international_plan,
        voice_mail_plan,
        number_vmail_messages,
        total_day_minutes,
        total_day_calls,
        total_day_charge,
        total_eve_minutes,
        total_eve_calls,
        total_eve_charge,
        total_night_minutes,
        total_night_calls,
        total_night_charge,
        total_intl_minutes,
        total_intl_calls,
        total_intl_charge,
        customer_service_calls
    ]

    # Load feature names
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    # Create DataFrame
    input_df = pd.DataFrame([features], columns=feature_names)

    # Load model
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    # Display result
    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is not likely to churn.")

    st.write(f"Churn Probability: {probability[0][1] * 100:.2f}%")
    st.write(f"Not Churn Probability: {probability[0][0] * 100:.2f}%")


