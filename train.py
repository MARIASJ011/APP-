import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\budget_data.csv")

# Encode categorical variables
encoders = {}
for col in ['Has_Debt', 'Owns_Asset', 'Outcome']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Save feature names for inference
with open("model/features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/encoder.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Evaluation
print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))
