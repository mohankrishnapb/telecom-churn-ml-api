import pandas as pd

# Load dataset
def load_data(path):
    df = pd.read_csv(path)
    return df

# Clean data
def clean_data(df):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Remove missing values
    df = df.dropna()
    
    # Drop unnecessary column
    df = df.drop("customerID", axis=1)
    
    return df

# Encode data
def encode_data(df):
    # Convert target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    # Convert categorical to numeric
    df = pd.get_dummies(df, drop_first=True)
    
    return df