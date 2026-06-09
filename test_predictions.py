import pandas as pd
import joblib
import json
import os

# Load data
df = pd.read_csv("customer_segmentation_processed.csv")

# Load models
models_dir = "models"
xgb = joblib.load(os.path.join(models_dir, "xgboost.pkl"))
with open(os.path.join(models_dir, "feature_columns.json"), "r") as f:
    feature_columns = json.load(f)

# Drop non-feature columns
X = df.drop(columns=[
    'Churn Label',
    'CustomerID',
    'Count',
    'Lat Long',
    'Latitude',
    'Longitude',
    'Churn Reason',
    'Churn Value',
    'Churn Score'
], errors='ignore')

# One-hot encode
X_encoded = pd.get_dummies(X)
# Align with training columns
X_final = X_encoded.reindex(columns=feature_columns, fill_value=0)

print("X_final shape:", X_final.shape)
print("feature_columns length:", len(feature_columns))

# Predict
probs = xgb.predict_proba(X_final)[:, 1]
print("Probabilities predicted successfully. Length:", len(probs))
print("Mean predicted churn probability:", probs.mean())
