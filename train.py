import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Optional: Prophet for forecasting
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False

# ------------------------------------
# Streamlit Page Setup
# ------------------------------------
st.set_page_config(page_title="💰 Finance Predictor", layout="wide")
st.title("💰 Personal Finance Tracker & Behavior Predictor")

# ------------------------------------
# Train model if not found
# ------------------------------------
@st.cache_resource
def load_or_train_model():
    if not os.path.exists("model.pkl"):
        st.warning("Model not found – training default model.")
        np.random.seed(42)
        size = 200

        income = np.random.normal(3500, 800, size).astype(int)
        expenses = income - np.random.normal(500, 400, size).astype(int)
        expenses = np.clip(expenses, 0, None)

        savings_ratio = (income - expenses) / income
        labels = (savings_ratio >= 0.2).astype(int)

        df = pd.DataFrame({
            "income": income,
            "expenses": expenses,
            "savings_behavior": labels
        })

        X = df[["income", "expenses"]]
        y = df["savings_behavior"]

        model = RandomForestClassifier()
        model.fit(X, y)

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    return model

model = load_or_train_model()
required_features = ["income", "expenses"]

# ------------------------------------
# Prediction Function
# ------------------------------------
def predict_behavior(df):
    for col in required_features:
        if col not in df.columns:
            df[col] = 0
    df = df[required_features]
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    return pd.DataFrame({
        "Prediction": preds,
        "Probability": probs
    })

# ------------------------------------
# SINGLE Prediction
# ------------------------------------
st.header("📍 Single Entry Prediction")
with st.form("single_form"):
    income = st.number_input("Monthly Income", min_value=0, value=3500)
    expenses = st.number_input("Monthly Expenses", min_value=0, value=2400)
    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([[income, expenses]], columns=required_features)
    result = predict_behavior(input_df)
    label = "Good" if result["Prediction"][0] == 1 else "Poor"
    st.success(f"Prediction: **{label} Saving Behavior** (Prob: {result['Probability'][0]:.2f})")

# ------------------------------------
# BATCH Prediction
# ------------------------------------
st.header("📤 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV with income, expenses, [date]", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())

    predictions = predict_behavior(df)
    output = df.copy()
    output["Prediction"] = predictions["Prediction"]
    output["Probability"] = predictions["Probability"]
    output["Label"] = output["Prediction"].map({0: "Poor", 1: "Good"})

    st.subheader("📊 Predictions")
    st.dataframe(output)

    # Download
    csv_data = output.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv_data, "predictions.csv", "text/csv")

    # Visualization: Count by class
    st.subheader("📈 Prediction Class Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=output, x="Label", ax=ax1)
    ax1.set_title("Saving Behavior Predictions")
    st.pyplot(fig1)

    # Scatterplot: Income vs Expenses
    st.subheader("📉 Income vs Expenses (Colored by Prediction)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=output, x="income", y="expenses", hue="Label", ax=ax2, palette="Set2")
    st.pyplot(fig2)

    # ------------------------------------
    # Forecasting
    # ------------------------------------
    st.header("📅 Forecast Future Income")

    if prophet_available and "date" in df.columns and "income" in df.columns:
        try:
            forecast_data = df[["date", "income"]].dropna()
            forecast_data["date"] = pd.to_datetime(forecast_data["date"])
            prophet_df = forecast_data.rename(columns={"date": "ds", "income": "y"})

            model_prophet = Prophet()
            model_prophet.fit(prophet_df)

            future = model_prophet.make_future_dataframe(periods=6, freq="M")
            forecast = model_prophet.predict(future)

            st.subheader("📈 Forecast (6 Months)")
            fig3 = model_prophet.plot(forecast)
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"Forecasting failed: {e}")
    elif not prophet_available:
        st.info("Install `prophet` to enable forecasting.")
    else:
        st.info("Upload must include 'date' and 'income' columns to forecast.")

# ------------------------------------
# Footer
# ------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, scikit-learn, and Prophet (optional)")
