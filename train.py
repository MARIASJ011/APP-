import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from io import BytesIO

# --------------------
# Configuration
# --------------------
st.set_page_config(page_title="💰 Finance Tracker & Predictor", layout="wide")
st.title("💰 Personal Finance Tracker & Forecasting App")

# --------------------
# Caching model loading
# --------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
expected_features = ["income", "expenses"]

# --------------------
# Prediction function
# --------------------
def predict_behavior(df):
    missing_cols = [col for col in expected_features if col not in df.columns]
    for col in missing_cols:
        df[col] = 0
    df = df[expected_features]
    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)
    return pd.DataFrame({
        "Prediction": preds,
        "Probability": probs
    })

# --------------------
# SINGLE RECORD PREDICTION
# --------------------
st.header("📍 Single Record Prediction")
with st.form("single_form"):
    income = st.number_input("Monthly Income", min_value=0, value=3000)
    expenses = st.number_input("Monthly Expenses", min_value=0, value=2000)
    submit = st.form_submit_button("Predict")

if submit:
    single_df = pd.DataFrame([[income, expenses]], columns=expected_features)
    result = predict_behavior(single_df)
    label = "Good Saving Behavior" if result["Prediction"][0] == 1 else "Poor Saving Behavior"
    st.success(f"Prediction: **{label}** (Probability: {result['Probability'][0]:.2f})")

# --------------------
# BATCH PREDICTION
# --------------------
st.header("📤 Batch Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload CSV (with income, expenses[, date])", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(batch_df.head())

    # Predict
    result_df = predict_behavior(batch_df)
    full_output = batch_df.copy()
    full_output["Prediction"] = result_df["Prediction"]
    full_output["Probability"] = result_df["Probability"]

    st.subheader("📊 Prediction Results")
    st.dataframe(full_output)

    # Download predictions
    csv_out = full_output.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv_out, "predictions.csv", "text/csv")

    # --------------------
    # Visualizations
    # --------------------
    st.subheader("📈 Prediction Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Prediction", data=full_output, ax=ax1)
    ax1.set_xticklabels(["Poor", "Good"])
    ax1.set_title("Saving Behavior Predictions")
    st.pyplot(fig1)

    st.subheader("📉 Income vs. Expenses (Colored by Prediction)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=full_output, x="income", y="expenses", hue="Prediction", palette="Set2", ax=ax2)
    st.pyplot(fig2)

    # --------------------
    # FORECASTING with Prophet
    # --------------------
    st.header("📅 Forecast Future Income (Optional)")

    if "date" in batch_df.columns and "income" in batch_df.columns:
        try:
            forecast_df = batch_df[["date", "income"]].dropna()
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])
            prophet_df = forecast_df.rename(columns={"date": "ds", "income": "y"})

            m = Prophet()
            m.fit(prophet_df)

            future = m.make_future_dataframe(periods=6, freq='M')
            forecast = m.predict(future)

            st.subheader("📈 Income Forecast (Next 6 Months)")
            fig3 = m.plot(forecast)
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"Forecasting failed: {e}")
    else:
        st.info("To use forecasting, your CSV must include 'date' and 'income' columns.")

# --------------------
# Footer
# --------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, scikit-learn, and Prophet")
