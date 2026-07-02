import streamlit as st
import pickle
import pandas as pd

st.header("📂 Bulk Customer Churn Prediction")
st.info(
    "Upload a CSV file containing the same feature columns used to train the model. "
    "The app will predict churn for every customer and allow you to download the results."
)
uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    bulk_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(bulk_df)

    # Convert Yes/No columns to 1/0
    bulk_df["International plan"] = bulk_df["International plan"].map(
        {"Yes": 1, "No": 0}
    )

    bulk_df["Voice mail plan"] = bulk_df["Voice mail plan"].map(
        {"Yes": 1, "No": 0}
    )

    # Load feature names
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    # Arrange columns correctly
    X_bulk = bulk_df[feature_names]

    # Load model
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predictions
    predictions = model.predict(X_bulk)
    probabilities = model.predict_proba(X_bulk)

    # Add results
    bulk_df["Prediction"] = predictions
    bulk_df["Prediction"] = bulk_df["Prediction"].map(
        {0: "No Churn", 1: "Churn"}
    )

    bulk_df["Churn Probability (%)"] = (
        probabilities[:, 1] * 100
    ).round(2)

    st.subheader("Prediction Results")
    st.dataframe(bulk_df)

    # Dashboard metrics
    total = len(bulk_df)
    churn = (bulk_df["Prediction"] == "Churn").sum()
    not_churn = total - churn

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Customers", total)
    c2.metric("Likely to Churn", churn)
    c3.metric("Not Likely to Churn", not_churn)

    # Download button
    csv = bulk_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name="customer_churn_predictions.csv",
        mime="text/csv"
    )
