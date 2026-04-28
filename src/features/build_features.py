import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.

    This function implement the core binary encodign logic that converts categorical features with exactly
    2 values into 0/1 integers. The mappings are deterministic and must be consistent between training and serving.
    """

    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # --- Deterministic Binary Mapping ---
    # Critical: These mapping are hardcoded in serving pipeline.

    # Yes/No mapping (most common pattern in telecom data)

    if valset == {"Yes", "No"}:
        return s.map({"Yes": 1, "No": 0}).astype("Int64")
    
    # Gender mapping (another common pattern)
    if valset == {"Male", "Female"}:
        return s.map({"Male": 1, "Female": 0}).astype("Int64")
    
    # ---Generic Binary MApping ---
    # For any other 2-category feature, use stable alphabetical ordering

    if len(vals) == 2:
        # Sort values to ensure consistent mapping across runs
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")
    
    # --- Non-Binary Features ---
    # Return unchanged  -will be handled by one-hot encoding.
    return s    

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply Complete Featue Engineering Pipeline for training data.
    This is the main feature engineering function that transforms raw customer data into 
    ML-ready features. The transformations must be exactly replicated in the serving pipeline to 
    ensure prediction accuracy.
    """

    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")

    # --- STEP1: Identify Feature Types ---
    # Find Categorical columns (Object type) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include="object").columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"Found {len(obj_cols)} categorical columns and {len(numeric_cols)} numeric columns.")

    # --- Step 2: Split Categorical by Cardinality ---
    # Binary Features (Exactly 2 unique values) get binary encoding.
    # Multi-category features (3 or more unique values) get one-hot encoding.

    binary_cols = [c for c in obj_cols if df[c].dropna().nunique()== 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    print(f"Binary Features: {len(binary_cols)} columns | Multi-category Features: {len(multi_cols)} columns.")

    if binary_cols:
        print(f"Binary Columns: {binary_cols}")
    if multi_cols:
        print(f"Multi-category Columns: {multi_cols}")

    # --- Step 3: Apply Binary Encoding ---
    # Convert 2-Category Features to 0/1 integers using deterministic mapping.
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))  # Ensure consistent string type for mapping
        print(f"{c}: {original_dtype} -> binary (0/1)")

    # --- Step 4: Apply One-Hot Encoding ---
    # XGBoost requires integer inputs, not boolean
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f" Converted {len(bool_cols)} boolean columns to integers: {bool_cols}")

    # --- Step 5 : One-Hot encoding for Multi-category Features ---
    # CRITICAL: drop_first=True prevents multicollinearity 
    if multi_cols:
        print(f"Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape

        # Apply one-hot encoding with drop_first=True to avoid multicollinearity
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)    

        new_features = df.shape[1] - original_shape[1] + len(multi_cols)  # New features added by one-hot encoding..
        print(f"Created {new_features} new features from {len(multi_cols)} categorical columns.")

        # ---Step 6: Data Type Cleanup ---
        # Convert nullable Integers (Int64) to standard integers for XGBoost.
        for c in binary_cols:
            if pd.api.types.is_integer_dtype(df[c]):
                df[c] = df[c].fillna(0).astype(int)  # Fill NaN with 0 for binary features
        
        print(f"Feature Engineering complete: {df.shape[1]} final features.")
        return df