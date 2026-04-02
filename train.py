from data_preprocessing import load_data, clean_data, encode_data
from model_training import train_model
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split
import joblib

# Load
df = load_data(r"C:\Users\HP\Desktop\CNN_PROJECT\telecom_chrun_project\archive (17)\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean
df = clean_data(df)

# Encode
df = encode_data(df)

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = train_model(X_train, y_train)

# Evaluate
evaluate_model(model, X_test, y_test)

# Save
joblib.dump(model, "model/churn_model.pkl")

print("\nModel saved successfully!")