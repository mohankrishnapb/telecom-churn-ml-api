from fastapi import FastAPI
import pandas as pd
import joblib
import os

app = FastAPI()

# Load model safely
model_path = "model/churn_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print("Warning: Model file not found!")

# Mapping dictionaries for categorical features
gender_map = {"Female": 0, "Male": 1}
internet_service_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_method_map = {  # example if you have more categorical fields
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Convert incoming JSON to DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical features
        if "gender" in df.columns:
            df["gender"] = df["gender"].map(gender_map)
        if "internet_service" in df.columns:
            df["internet_service"] = df["internet_service"].map(internet_service_map)
        if "contract" in df.columns:
            df["contract"] = df["contract"].map(contract_map)
        if "payment_method" in df.columns:
            df["payment_method"] = df["payment_method"].map(payment_method_map)
        
        # Make sure all values are numeric
        df = df.astype(float)
        
        # Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return {
            "churn": int(prediction),
            "probability": float(probability)
        }
    
    except Exception as e:
        return {"error": str(e)}