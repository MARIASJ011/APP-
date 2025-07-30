import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file):
    df = pd.read_csv(file)
    return df

def train_model(data):
    data = data.copy()
    le = LabelEncoder()
    data['Has_Debt'] = le.fit_transform(data['Has_Debt'])
    data['Owns_Asset'] = le.fit_transform(data['Owns_Asset'])
    data['Outcome'] = le.fit_transform(data['Outcome'])

    features = ['Monthly_Income', 'Monthly_Expenses', 'Savings', 'Has_Debt', 'Owns_Asset']
    target = 'Outcome'

    model = RandomForestClassifier()
    model.fit(data[features], data[target])

    return model, le, features

def forecast_savings(data):
    model = LinearRegression()
    model.fit(data[['Monthly_Income', 'Monthly_Expenses']], data['Savings'])
    return model

def main():
    st.set_page_config(page_title="Personal Finance Tracker", layout="centered")
    st.title("💰 Personal Finance Tracker with Forecasting")

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("🔍 Preview of Uploaded Data", df.head())

        model, le, feature_names = train_model(df)

        st.subheader("📊 Make a Prediction")
        income = st.number_input("Monthly Income", min_value=10000, max_value=200000)
        expenses = st.number_input("Monthly Expenses", min_value=5000, max_value=150000)
        savings = st.number_input("Current Savings", min_value=0, max_value=100000)
        debt = st.radio("Has Debt?", ["Yes", "No"])
        asset = st.radio("Owns Asset?", ["Yes", "No"])

        input_df = pd.DataFrame({
            'Monthly_Income': [income],
            'Monthly_Expenses': [expenses],
            'Savings': [savings],
            'Has_Debt': le.transform([debt]),
            'Owns_Asset': le.transform([asset])
        })

        prediction = model.predict(input_df[feature_names])[0]
        outcome = le.inverse_transform([prediction])[0]
        st.success(f"💡 Prediction: Your financial status is likely to be **{outcome}**.")

        st.subheader("📈 Prediction Distribution")
        df['Prediction'] = model.predict(df[feature_names])
        df['Prediction_Label'] = le.inverse_transform(df['Prediction'])
        sns.countplot(x="Prediction_Label", data=df)
        st.pyplot(plt)

        st.subheader("📉 Forecast Savings Based on Future Plans")
        savings_model = forecast_savings(df)
        future_income = st.slider("Future Income", 20000, 150000, 50000)
        future_expense = st.slider("Future Expenses", 5000, 120000, 25000)
        predicted_savings = savings_model.predict([[future_income, future_expense]])
        st.success(f"🧮 Predicted Future Savings: ₹{predicted_savings[0]:,.2f}")

if __name__ == "__main__":
    main()
