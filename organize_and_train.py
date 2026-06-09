import os
import shutil
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

def main():
    workspace = r"c:\Users\bhard\OneDrive\Desktop\customer"
    project_dir = os.path.join(workspace, "Customer-Churn-Prediction")
    
    # 1. Create directories
    dirs = ["data", "notebooks", "models", "images"]
    for d in dirs:
        os.makedirs(os.path.join(project_dir, d), exist_ok=True)
    print("Project directories created.")

    # 2. Move existing files if present in the workspace root
    files_to_move = {
        "Telco_customer_churn.xlsx": "data",
        "customer_segmentation_processed.csv": "data",
        "06_Final_Insights_Report.ipynb": "notebooks",
        "07_Churn_Prediction.ipynb": "notebooks"
    }
    
    for filename, folder in files_to_move.items():
        src = os.path.join(workspace, filename)
        dst = os.path.join(project_dir, folder, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {filename} to {folder}/")
        else:
            if os.path.exists(dst):
                print(f"{filename} already in {folder}/")
            else:
                print(f"Warning: {filename} not found in root or destination!")

    # 3. Load processed data for model training
    csv_path = os.path.join(project_dir, "data", "customer_segmentation_processed.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Processed data file not found at {csv_path}!")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded processed data. Shape: {df.shape}")

    # 4. Fit and save K-Means and Scaler
    segment_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']
    segment_df = df[segment_cols].copy()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(segment_df)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster_pred'] = kmeans.fit_predict(scaled_data)
    
    # Verify cluster alignment with the existing Segment/Cluster labels
    # Let's map cluster IDs to segment names based on centroids
    # VIP: High tenure, high monthly, high total, high CLTV
    # Loyal: High tenure, low monthly, medium total, high CLTV
    # At-Risk: Low-mid tenure, mid monthly, low-mid total, low CLTV
    # New: Low tenure, mid monthly, low total, high CLTV
    centroids = kmeans.cluster_centers_
    # Let's run a check to associate the K-Means cluster IDs with their Segment names in training data
    cluster_to_segment = {}
    for cluster_id in range(4):
        mode_segment = df[df['Cluster_pred'] == cluster_id]['Segment'].mode()[0]
        cluster_to_segment[cluster_id] = mode_segment
        print(f"KMeans Cluster {cluster_id} mapped to Segment: {mode_segment}")
        
    # Save the scaler, kmeans, and the mapping
    models_dir = os.path.join(project_dir, "models")
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans.pkl"))
    with open(os.path.join(models_dir, "cluster_mapping.json"), "w") as f:
        json.dump({str(k): v for k, v in cluster_to_segment.items()}, f)
    print("Saved Scaler, KMeans, and Cluster mapping to models/.")

    # 5. Prepare data for Classification Models
    # Map target column Churn Label to binary values
    df['Churn Label'] = df['Churn Label'].map({'No': 0, 'Yes': 1})
    
    # Define features X and target y
    X = df.drop(columns=[
        'Churn Label',
        'CustomerID',
        'Count',
        'Lat Long',
        'Latitude',
        'Longitude',
        'Churn Reason',
        'Churn Value',
        'Churn Score',
        'Cluster_pred' # drop the prediction column we just added for check
    ])
    y = df['Churn Label']
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_cols = X_encoded.columns.tolist()
    
    # Save feature columns list
    with open(os.path.join(models_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    print(f"Saved {len(feature_cols)} feature columns to models/feature_columns.json.")

    # Split data (matching the notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Train and save models
    # A. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)
    joblib.dump(lr, os.path.join(models_dir, "logistic.pkl"))
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # B. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    joblib.dump(rf, os.path.join(models_dir, "random_forest.pkl"))
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # C. XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    xgb_acc = xgb.score(X_test, y_test)
    joblib.dump(xgb, os.path.join(models_dir, "xgboost.pkl"))
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")

    print("All models trained and saved successfully.")

if __name__ == "__main__":
    main()
