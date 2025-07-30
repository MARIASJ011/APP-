import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------ #
# PART 1: DATA & TRAINING  #
# ------------------------ #

@st.cache_resource
def train_model():
    # Generate synthetic dataset
    n = 500
    df = pd.DataFrame({
        'Monthly_Income': np.random.randint(20000, 150000, n),
        'Monthly_Expenses': np.random.randint(5000, 120000, n),
        'Savings': np.random.randint(0, 50000, n),
        'Has_Debt': np.random.choice(['Yes', 'No'], n),
        'Owns_Asset': np.random.choice(['Yes', 'No'], n),
        'Outcome': np.random.choice(['Stable', 'Unstable'], n)
    })

    # Label encoding
    encoders = {}
    for col in ['Has_Debt', 'Owns_Asset', 'Outcome']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save files
    with open("model.pkl", "wb") as f: pickle.dump(model, f)
    with open("encoder.pkl", "wb") as f: pickle.dump(encoders, f)
    with open("features.pkl", "wb") as f: pickle.dump(features, f)

    return model, encoders, features

# ------------------------ #
# PART 2: LOAD RESOURCES   #
# ------------------------ #

@st.cache_resource
def load_resources():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("encoder.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open("features.pkl", "rb") as f:
            features = pickle.load(f)
    except:
        model, encoders, features = train_model()
    return model, encoders, features

model, encoders, features = load_resources()

# ------------------------ #
# PART 3: STREAMLIT UI     #
# ------------------------ #

st.title("💰 Personal Finance Tracker - ML Powered Outcome Prediction")

mode = st.sidebar.radio("Select Prediction Mode", ["Single Prediction", "Batch Prediction"])

def preprocess_input(data, encoders, features):
    df = data.copy()
    for col in ['Has_Debt', 'Owns_Asset']:
        if col in df.columns:
            df[col] = df[col].fillna('No')
            df[col] = encoders[col].transform(df[col])
    return df[features]

if mode == "Single Prediction":
    st.subheader("📄 Enter Your Financial Details")

    income = st.number_input("Monthly Income", value=50000)
    expenses = st.number_input("Monthly Expenses", value=25000)
    savings = st.number_input("Monthly Savings", value=10000)
    debt = st.selectbox("Has Debt?", ['Yes', 'No'])
    asset = st.selectbox("Owns Asset?", ['Yes', 'No'])

    if st.button("Predict Outcome"):
        input_data = pd.DataFrame({
            'Monthly_Income': [income],
            'Monthly_Expenses': [expenses],
            'Savings': [savings],
            'Has_Debt': [debt],
            'Owns_Asset': [asset]
        })
        processed = preprocess_input(input_data, encoders, features)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed).max()
        label = encoders['Outcome'].inverse_transform([prediction])[0]

        st.success(f"🧾 Prediction: **{label}** with probability {probability:.2f}")

elif mode == "Batch Prediction":
    st.subheader("📁 Upload a CSV File")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        try:
            data = pd.read_csv(file)
            st.write("📋 Uploaded Data", data.head())

            # Validate required columns
            required_cols = set(features)
            missing_cols = required_cols - set(data.columns)

            if missing_cols:
                st.error(f"❌ Uploaded file is missing required columns: {', '.join(missing_cols)}")
                st.info("✅ Required columns are: " + ", ".join(required_cols))
            else:
                processed = preprocess_input(data, encoders, features)
                predictions = model.predict(processed)
                predicted_labels = encoders['Outcome'].inverse_transform(predictions)
                data["Prediction"] = predicted_labels
                st.write("📊 Prediction Results:", data)

                # Download button
                csv = data.to_csv(index=False).encode()
                st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

                # Plot prediction distribution
                st.subheader("📈 Prediction Distribution")
                fig, ax = plt.subplots()
                sns.countplot(x="Prediction", data=data, ax=ax)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction:\n\n{str(e)}")

            # Visual Summary
            st.subheader("📈 Prediction Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x="Prediction", data=data, ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

