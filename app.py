import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders
@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/encoder.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("model/features.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, encoders, feature_names

model, encoders, feature_names = load_model()

# App title
st.title("💰 Personal Finance Tracker - Outcome Prediction")

st.sidebar.header("Choose Prediction Mode")
mode = st.sidebar.radio("Select mode", ["Single Prediction", "Batch Prediction"])

def preprocess(df):
    df = df.copy()
    for col in ['Has_Debt', 'Owns_Asset']:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col].fillna('No'))
    df = df[feature_names]
    return df

if mode == "Single Prediction":
    st.subheader("📄 Enter Details for a Single Prediction")

    income = st.number_input("Monthly Income", value=50000)
    expenses = st.number_input("Monthly Expenses", value=30000)
    savings = st.number_input("Savings", value=10000)
    debt = st.selectbox("Has Debt?", ['Yes', 'No'])
    asset = st.selectbox("Owns Asset?", ['Yes', 'No'])

    if st.button("Predict"):
        input_df = pd.DataFrame({
            "Monthly_Income": [income],
            "Monthly_Expenses": [expenses],
            "Savings": [savings],
            "Has_Debt": [debt],
            "Owns_Asset": [asset]
        })
        X_input = preprocess(input_df)
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input).max()
        label = encoders['Outcome'].inverse_transform([pred])[0]

        st.success(f"🔍 Prediction: **{label}** with probability {prob:.2f}")

elif mode == "Batch Prediction":
    st.subheader("📁 Upload CSV File for Batch Prediction")

    file = st.file_uploader("Upload a CSV", type=["csv"])
    if file:
        data = pd.read_csv(file)
        st.write("✅ Uploaded Data", data.head())

        try:
            processed = preprocess(data)
            predictions = model.predict(processed)
            labels = encoders['Outcome'].inverse_transform(predictions)
            data["Prediction"] = labels
            st.write("🔍 Predictions", data)

            # Download option
            csv = data.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

            # Visualizations
            st.subheader("📊 Visualizations")
            fig, ax = plt.subplots()
            sns.countplot(x="Prediction", data=data, ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
