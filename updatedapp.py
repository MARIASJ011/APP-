import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data(file):
    return pd.read_csv(file)

def preprocess_data(data):
    le = LabelEncoder()
    for col in ['Has_Debt', 'Owns_Asset', 'Outcome']:
        if col in data.columns:
            data[col] = le.fit_transform(data[col])
    return data, le

def train_classifier_model(data):
    features = ['Monthly_Income', 'Monthly_Expenses', 'Savings', 'Has_Debt', 'Owns_Asset']
    target = 'Outcome'
    clf = RandomForestClassifier()
    clf.fit(data[features], data[target])
    return clf, features

def train_savings_forecast_model(data):
    reg = LinearRegression()
    reg.fit(data[['Monthly_Income', 'Monthly_Expenses']], data['Savings'])
    return reg

def single_prediction_form(le, clf, features):
    st.subheader("🧠 Single Prediction Input")

    income = st.number_input("Monthly Income (₹)", 1000, 200000, 50000)
    expenses = st.number_input("Monthly Expenses (₹)", 500, 150000, 20000)
    savings = st.number_input("Current Savings (₹)", 0, 1000000, 10000)
    debt = st.selectbox("Has Debt?", ["Yes", "No"])
    asset = st.selectbox("Owns Asset?", ["Yes", "No"])

    df_input = pd.DataFrame([{
        "Monthly_Income": income,
        "Monthly_Expenses": expenses,
        "Savings": savings,
        "Has_Debt": le.transform([debt])[0],
        "Owns_Asset": le.transform([asset])[0]
    }])

    if st.button("🔮 Predict Financial Status"):
        pred = clf.predict(df_input[features])[0]
        label = le.inverse_transform([pred])[0]
        st.success(f"Prediction: **{label}**")

def batch_prediction(df, clf, le, features):
    df['Prediction'] = clf.predict(df[features])
    df['Prediction_Label'] = le.inverse_transform(df['Prediction'])
    return df

def plot_prediction_distribution(df):
    st.subheader("📊 Prediction Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Prediction_Label', data=df, ax=ax)
    st.pyplot(fig)

def forecast_savings_form(reg):
    st.subheader("📈 Forecast Your Future Savings")

    income = st.slider("Expected Monthly Income (₹)", 10000, 200000, 50000, step=1000)
    expenses = st.slider("Expected Monthly Expenses (₹)", 5000, 150000, 25000, step=1000)

    pred = reg.predict([[income, expenses]])
    st.success(f"Estimated Future Savings: ₹{pred[0]:,.2f}")

def main():
    st.set_page_config(page_title="💰 Personal Finance Tracker", layout="centered")
    st.title("💼 Personal Finance Tracker + Prediction + Forecast")

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("🔎 Preview of Uploaded Data", df.head())

        processed_df, le = preprocess_data(df)
        clf, features = train_classifier_model(processed_df)
        reg = train_savings_forecast_model(processed_df)

        # Single prediction
        single_prediction_form(le, clf, features)

        # Batch prediction
        st.subheader("📁 Batch Predictions on Uploaded Data")
        df_with_preds = batch_prediction(processed_df, clf, le, features)
        st.write(df_with_preds[['Monthly_Income', 'Monthly_Expenses', 'Savings', 'Prediction_Label']].head())

        # Plot
        plot_prediction_distribution(df_with_preds)

        # Forecasting
        forecast_savings_form(reg)

    else:
        st.warning("⚠️ Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
