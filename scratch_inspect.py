import pandas as pd
df = pd.read_csv("customer_segmentation_processed.csv")
print("Columns:")
print(df.columns.tolist())
print("\nUnique values for some categorical fields:")
for col in ['Segment', 'Contract', 'Internet Service', 'Tech Support', 'Online Security']:
    if col in df.columns:
        print(f"{col}: {df[col].unique().tolist()}")
    else:
        print(f"Warning: {col} not in columns")

if 'Tenure_Group' in df.columns:
    print(f"Tenure_Group unique values: {df['Tenure_Group'].unique().tolist()}")
else:
    print("Tenure_Group not present in CSV")
