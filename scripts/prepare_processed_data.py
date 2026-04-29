import os, sys
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = os.path.join("data", "raw", "customer_churn.csv")
OUT = os.path.join("data", "processed", "telco_churn_processed.csv")

# Load data
df = pd.read_csv(RAW)

# Preprocess data
df = preprocess_data(df, target_col="Churn")

# Ensure target is 0/1 only if still object
if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"Yes": 1, "No": 0}).astype("Int64")  

# Sanity Checks
assert df["Churn"].isna().sum() == 0, "Target column 'Churn' contains null values after preprocessing."
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1 after preprocess"

# Build features
df_processed = build_features(df, target_col="Churn")

# Save
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)
print(f"Processed data saved to {OUT}. Shape: {df_processed.shape}")
